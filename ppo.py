import os
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import warnings
import logging
from collections import deque

# Import enhanced environment
from aquacropgymnasium.env import EnhancedMaize

# Set up prettier plots (compatible with older versions)
try:
    plt.style.use('seaborn-whitegrid')  # For older versions
except:
    try:
        plt.style.use('ggplot')  # Alternative style
    except:
        pass  # Fall back to default style
sns.set_context("notebook", font_scale=1.2)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

@dataclass
class Args:
    exp_name: str = "enhanced_maize_ppo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "enhanced_maize_ppo"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.3
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.05
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
    # Custom environment arguments
    year1: int = 1982
    """starting year for training (inclusive)"""
    year2: int = 2007
    """ending year for training (inclusive)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RunningMeanStd:
    """Tracks the running mean and std of observation features for normalization"""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        self.var = (self.var * self.count + batch_var * batch_count +
                   delta**2 * self.count * batch_count / tot_count) / tot_count
        self.mean = new_mean
        self.count = tot_count
    
    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_size = np.array(envs.single_observation_space.shape).prod()
        
        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01),
        )
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def make_env(year1, year2, seed, idx):
    """Create a wrapped EnhancedMaize environment"""
    def thunk():
        env = EnhancedMaize(mode='train', year1=year1, year2=year2)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env
    return thunk


class EpisodeStats:
    def __init__(self, output_dir):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_yields = []
        self.episode_irrigation = []
        self.episode_wue = []  # Water Use Efficiency
        self.episode_count = 0
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Keep a deque for recent performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.recent_yields = deque(maxlen=100)
        self.recent_irrigation = deque(maxlen=100)
        
    def add_episode(self, reward, length, dry_yield, irrigation, wue=None):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_yields.append(dry_yield)
        self.episode_irrigation.append(irrigation)
        if wue is not None:
            self.episode_wue.append(wue)
        
        self.recent_rewards.append(reward)
        self.recent_yields.append(dry_yield)
        self.recent_irrigation.append(irrigation)
        
        self.episode_count += 1
    
    def get_recent_stats(self):
        """Return mean and std of recent episodes"""
        if len(self.recent_rewards) == 0:
            return (0, 0), (0, 0), (0, 0)
        
        reward_mean = np.mean(self.recent_rewards)
        reward_std = np.std(self.recent_rewards)
        
        yield_mean = np.mean(self.recent_yields)
        yield_std = np.std(self.recent_yields)
        
        irr_mean = np.mean(self.recent_irrigation)
        irr_std = np.std(self.recent_irrigation)
        
        return (reward_mean, reward_std), (yield_mean, yield_std), (irr_mean, irr_std)
    
    def plot_stats(self):
        """Simple plotting function to avoid dimension errors"""
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot rewards (raw data only)
        plt.figure(figsize=(12, 7))
        episodes = range(1, self.episode_count + 1)
        plt.plot(episodes, self.episode_rewards, color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'rewards.png'))
        plt.close()
        
        # Plot yields (raw data only)
        plt.figure(figsize=(12, 7))
        plt.plot(episodes, self.episode_yields, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Dry Yield (tonne/ha)')
        plt.title('Crop Yields')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'yields.png'))
        plt.close()
        
        # Plot irrigation (raw data only)
        plt.figure(figsize=(12, 7))
        plt.plot(episodes, self.episode_irrigation, color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Total Irrigation (mm)')
        plt.title('Irrigation Applied')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'irrigation.png'))
        plt.close()
        
        # Plot Water Use Efficiency if available
        if self.episode_wue:
            plt.figure(figsize=(12, 7))
            wue_episodes = range(1, len(self.episode_wue) + 1)
            plt.plot(wue_episodes, self.episode_wue, color='purple')
            plt.xlabel('Episode')
            plt.ylabel('Water Use Efficiency (yield/irrigation)')
            plt.title('Water Use Efficiency')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, 'water_use_efficiency.png'))
            plt.close()

        # Create a summary plot with final performance
        try:
            # Get the last 100 episodes (or fewer if not available)
            recent_count = min(100, self.episode_count)
            if recent_count > 0:
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 2, 1)
                recent_rewards = self.episode_rewards[-recent_count:]
                plt.hist(recent_rewards, bins=10, color='blue', alpha=0.7)
                plt.axvline(np.mean(recent_rewards), color='red', linestyle='dashed', 
                        linewidth=2, label=f'Mean: {np.mean(recent_rewards):.2f}')
                plt.title('Recent Rewards')
                plt.legend()
                
                plt.subplot(2, 2, 2)
                recent_yields = self.episode_yields[-recent_count:]
                plt.hist(recent_yields, bins=10, color='green', alpha=0.7)
                plt.axvline(np.mean(recent_yields), color='red', linestyle='dashed',
                        linewidth=2, label=f'Mean: {np.mean(recent_yields):.2f}')
                plt.title('Recent Yields')
                plt.legend()
                
                plt.subplot(2, 2, 3)
                recent_irrigation = self.episode_irrigation[-recent_count:]
                plt.hist(recent_irrigation, bins=10, color='blue', alpha=0.7)
                plt.axvline(np.mean(recent_irrigation), color='red', linestyle='dashed',
                        linewidth=2, label=f'Mean: {np.mean(recent_irrigation):.2f}')
                plt.title('Recent Irrigation')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'performance_summary.png'))
                plt.close()
                
                # Create correlation heatmap
                plt.figure(figsize=(10, 8))
                
                # Create correlation matrix
                data = {
                    'Reward': self.episode_rewards[-recent_count:],
                    'Yield': self.episode_yields[-recent_count:],
                    'Irrigation': self.episode_irrigation[-recent_count:]
                }
                
                if self.episode_wue and len(self.episode_wue) >= recent_count:
                    data['WUE'] = self.episode_wue[-recent_count:]
                    
                # Create correlation matrix
                corr_data = np.array([data[k] for k in data]).T
                corr_matrix = np.corrcoef(corr_data.T)
                
                # Plot heatmap
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                          xticklabels=list(data.keys()), yticklabels=list(data.keys()),
                          vmin=-1, vmax=1)
                plt.title('Correlation Between Metrics', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Could not create summary plot: {e}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Create output directory
    output_dir = f"runs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logger
    writer = SummaryWriter(output_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.year1, args.year2, args.seed, i) for i in range(args.num_envs)]
    )
    
    # Initialize statistics tracking
    stats = EpisodeStats(output_dir)
    
    # Initialize observation normalization
    obs_rms = RunningMeanStd(shape=envs.single_observation_space.shape)
    
    # Create agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Initialize storage for episode information
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Episodic tracking across environments
    ep_rewards = [0] * args.num_envs
    ep_lengths = [0] * args.num_envs
    episode_count = 0
    
    # Initialize variables for tracking environment-specific info
    env_info = [{
        'yields': [],
        'irrigation': [],
        'wue': []
    } for _ in range(args.num_envs)]
    
    # Print start message
    print(f"Starting training on Enhanced Maize environment with PPO")
    print(f"Years range: {args.year1}-{args.year2}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Start timing
    global_step = 0
    start_time = time.time()
    
    # Reset environments and get initial observations
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # Main training loop
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the learning rate if requested
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Update running mean/std for observation normalization
        obs_rms.update(next_obs.cpu().numpy())
        
        # Collect rollout
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            # Store current observation and done flag
            obs[step] = next_obs
            dones[step] = next_done
            
            # Update episode lengths
            for env_idx in range(args.num_envs):
                if not next_done[env_idx]:
                    ep_lengths[env_idx] += 1
            
            # Get action and value from policy
            with torch.no_grad():
                # Normalize observations before passing to agent
                normalized_obs = torch.tensor(obs_rms.normalize(next_obs.cpu().numpy())).float().to(device)
                action, logprob, _, value = agent.get_action_and_value(normalized_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            # Execute action in environments
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.tensor(np.logical_or(terminations, truncations)).to(device)
            processed_reward = np.tanh(np.array(reward) / 1000.0)
            rewards[step] = torch.tensor(processed_reward).to(device)
            
            # Track episode rewards
            for env_idx in range(args.num_envs):
                ep_rewards[env_idx] += reward[env_idx]
            
            # Convert next_obs to tensor and store
            next_obs = torch.tensor(next_obs).to(device)
            
            # Handle episode terminations
            for env_idx in range(args.num_envs):
                if next_done[env_idx].item():
                    # Extract end-of-episode information
                    if 'final_info' in infos:
                        info = infos['final_info'][env_idx]
                    else:
                        info = infos[env_idx]
                    
                    # Get episode metrics
                    dry_yield = info.get('dry_yield', 0.0)
                    total_irrigation = info.get('total_irrigation', 0.0)
                    wue = info.get('water_use_efficiency', 0.0) if total_irrigation > 0 else 0.0
                    
                    # Store metrics for this environment
                    env_info[env_idx]['yields'].append(dry_yield)
                    env_info[env_idx]['irrigation'].append(total_irrigation)
                    env_info[env_idx]['wue'].append(wue)
                    
                    # Log to TensorBoard
                    writer.add_scalar("charts/episodic_return", ep_rewards[env_idx], episode_count)
                    writer.add_scalar("charts/episodic_length", ep_lengths[env_idx], episode_count)
                    writer.add_scalar("charts/dry_yield", dry_yield, episode_count)
                    writer.add_scalar("charts/total_irrigation", total_irrigation, episode_count)
                    if wue > 0:
                        writer.add_scalar("charts/water_use_efficiency", wue, episode_count)
                    
                    # Add to episode stats
                    stats.add_episode(
                        ep_rewards[env_idx], 
                        ep_lengths[env_idx], 
                        dry_yield, 
                        total_irrigation, 
                        wue
                    )
                    
                    # Print progress
                    print(f"Episode {episode_count} completed after {ep_lengths[env_idx]} steps")
                    print(f"  Reward: {ep_rewards[env_idx]:.2f}, Yield: {dry_yield:.2f}, Irrigation: {total_irrigation:.2f}")
                    if wue > 0:
                        print(f"  Water Use Efficiency: {wue:.4f}")
                    
                    # Get recent performance stats
                    if len(stats.recent_rewards) >= 10:
                        (reward_mean, reward_std), (yield_mean, yield_std), (irr_mean, irr_std) = stats.get_recent_stats()
                        print(f"  Recent performance (last 100 episodes):")
                        print(f"    Reward: {reward_mean:.2f} ± {reward_std:.2f}")
                        print(f"    Yield: {yield_mean:.2f} ± {yield_std:.2f}")
                        print(f"    Irrigation: {irr_mean:.2f} ± {irr_std:.2f}")
                    
                    # Reset episode tracking for this environment
                    ep_rewards[env_idx] = 0
                    ep_lengths[env_idx] = 0
                    episode_count += 1
            
        # Compute advantages and returns
        with torch.no_grad():
            next_value = agent.get_value(
                torch.tensor(obs_rms.normalize(next_obs.cpu().numpy())).float().to(device)
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # Flatten the batch for policy update
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Normalize observations for training
        b_obs = torch.tensor(obs_rms.normalize(b_obs.cpu().numpy())).float().to(device)
        
        # Optimize policy and value networks
        clipfracs = []
        
        # Iterate over optimization epochs
        for epoch in range(args.update_epochs):
            # Create random indices for minibatches
            indices = np.random.permutation(args.batch_size)
            
            # Iterate over minibatches
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_indices = indices[start:end]
                
                # Get new log probs, value estimates, and entropy
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_indices], b_actions[mb_indices]
                )
                logratio = newlogprob - b_logprobs[mb_indices]
                ratio = logratio.exp()
                
                # Calculate KL divergence
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                # Normalize advantages
                mb_advantages = b_advantages[mb_indices]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss with clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss with optional clipping
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_indices]) ** 2
                    v_clipped = b_values[mb_indices] + torch.clamp(
                        newvalue - b_values[mb_indices], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_indices]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Combined loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            
            # Break early if KL divergence threshold exceeded
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # Calculate explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Log metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        # Print training speed
        if iteration % 10 == 0:
            current_sps = int(global_step / (time.time() - start_time))
            print(f"Iteration {iteration}/{args.num_iterations}, SPS: {current_sps}")
            writer.add_scalar("charts/SPS", current_sps, global_step)
        
        # Save intermediate model every 100 iterations
        if iteration % 100 == 0 and args.save_model:
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_iter{iteration}.pt")
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iteration,
                'obs_rms_mean': obs_rms.mean,
                'obs_rms_var': obs_rms.var,
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    # Plot statistics
    stats.plot_stats()
    
    # Save final model
    if args.save_model:
        # Save full model
        model_path = os.path.join(output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': agent.state_dict(),
            'obs_rms_mean': obs_rms.mean,
            'obs_rms_var': obs_rms.var,
        }, model_path)
        print(f"Final model saved to {model_path}")
    
    # Record final metrics
    final_time = time.time() - start_time
    final_sps = global_step / final_time
    
    print(f"\nTraining completed:")
    print(f"  Total episodes: {episode_count}")
    print(f"  Total time: {final_time:.2f} seconds")
    print(f"  Average steps per second: {final_sps:.2f}")
    
    # Save final metrics
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(f"Total episodes: {episode_count}\n")
        f.write(f"Total time: {final_time:.2f} seconds\n")
        f.write(f"Average steps per second: {final_sps:.2f}\n")
        
        # Add recent performance stats
        (reward_mean, reward_std), (yield_mean, yield_std), (irr_mean, irr_std) = stats.get_recent_stats()
        f.write(f"\nRecent performance (last 100 episodes):\n")
        f.write(f"  Reward: {reward_mean:.2f} ± {reward_std:.2f}\n")
        f.write(f"  Yield: {yield_mean:.2f} ± {yield_std:.2f}\n")
        f.write(f"  Irrigation: {irr_mean:.2f} ± {irr_std:.2f}\n")
    
    # Clean up
    envs.close()
    writer.close()