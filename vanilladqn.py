import os
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from collections import deque

# Import your enhanced environment
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
    exp_name: str = "enhanced_maize_dqn"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "enhanced_maize_dqn"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the target network update rate (for soft updates)"""
    target_network_frequency: int = 1
    """update target network every n steps (1 for soft updates)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 0.05
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.3
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    
    # Custom environment arguments
    year1: int = 1982
    """starting year for training (inclusive)"""
    year2: int = 2007
    """ending year for training (inclusive)"""


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


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        obs_shape = observation_space.shape
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, info=None):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array([action]).copy()
        self.rewards[self.pos] = np.array([reward]).copy()
        self.dones[self.pos] = np.array([done]).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        indices = np.random.choice(upper_bound, batch_size, replace=False)

        class Batch:
            def __init__(self, observations, actions, next_observations, dones, rewards):
                self.observations = observations
                self.actions = actions
                self.next_observations = next_observations
                self.dones = dones
                self.rewards = rewards

        return Batch(
            torch.as_tensor(self.observations[indices]).float().to(self.device),
            torch.as_tensor(self.actions[indices]).long().to(self.device),
            torch.as_tensor(self.next_observations[indices]).float().to(self.device),
            torch.as_tensor(self.dones[indices]).float().to(self.device),
            torch.as_tensor(self.rewards[indices]).float().to(self.device),
        )



class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

    
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def smooth_curve(points, window_size=10):
    """Apply smoothing to a curve with proper handling of window edges"""
    kernel = np.ones(window_size) / window_size
    return np.convolve(points, kernel, mode='valid')


class EpisodeStats:
    def __init__(self, output_dir):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_yields = []
        self.episode_irrigation = []
        self.episode_wue = []  # Water Use Efficiency
        self.td_losses = []
        self.q_values = []
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
    
    def add_training_info(self, td_loss, q_value):
        self.td_losses.append(td_loss)
        self.q_values.append(q_value)
    
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
        """Ultra-simple plotting function with no smoothing to avoid any dimension errors"""
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
        except Exception as e:
            print(f"Could not create summary plot: {e}")


    def plot_training_metrics(self):
        if not self.td_losses:  # Skip if no training data collected
            return
        
        # Create directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
            
        # Plot TD Loss with better styling
        plt.figure(figsize=(12, 7))
        plt.plot(self.td_losses, alpha=0.3, color='crimson', label='TD Loss')
        
        # Add moving average with larger window
        if len(self.td_losses) >= 100:
            window_size = 100
            smoothed = smooth_curve(self.td_losses, window_size)
            smooth_x = np.arange(window_size/2, len(self.td_losses) - window_size/2 + 1)
            plt.plot(smooth_x, smoothed, 
                     color='darkred', linewidth=2.5, 
                     label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Training Updates', fontsize=14)
        plt.ylabel('TD Loss', fontsize=14)
        plt.title('TD Loss During Training', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'td_loss.png'), dpi=300)
        plt.close()
        
        # Plot Q-values with better styling
        plt.figure(figsize=(12, 7))
        plt.plot(self.q_values, alpha=0.3, color='teal', label='Mean Q-Value')
        
        if len(self.q_values) >= 100:
            window_size = 100
            smoothed = smooth_curve(self.q_values, window_size)
            smooth_x = np.arange(window_size/2, len(self.q_values) - window_size/2 + 1)
            plt.plot(smooth_x, smoothed, 
                     color='darkcyan', linewidth=2.5, 
                     label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Training Updates', fontsize=14)
        plt.ylabel('Mean Q-Value', fontsize=14)
        plt.title('Q-Value Magnitude During Training', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'q_values.png'), dpi=300)
        plt.close()
        
        # Create correlation heatmap if enough data
        if self.episode_count > 10:
            plt.figure(figsize=(10, 8))
            
            # Create correlation matrix
            data = {
                'Reward': self.episode_rewards,
                'Yield': self.episode_yields,
                'Irrigation': self.episode_irrigation
            }
            
            if self.episode_wue:
                data['WUE'] = self.episode_wue
                
            # Use only the data points where all values are available
            min_len = min(len(v) for v in data.values())
            for k in data:
                data[k] = data[k][:min_len]
                
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


def make_env(year1, year2, seed):
    """Create a wrapped, monitored Enhanced Maize environment"""
    def thunk():
        env = EnhancedMaize(mode='train', year1=year1, year2=year2)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return thunk


def process_reward(reward):
    """Process raw rewards for better stability"""
    # Use tanh to bound rewards without clipping information
    return np.tanh(reward / 1000.0)


if __name__ == "__main__":
    args = tyro.cli(Args)
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
    
    # Create environment
    env = make_env(args.year1, args.year2, args.seed)()
    stats = EpisodeStats(output_dir)
    
    # Initialize observation normalization
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    
    # Create networks
    q_network = QNetwork(np.prod(env.observation_space.shape), env.action_space.n).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(np.prod(env.observation_space.shape), env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device
    )

    
    # Start timer
    start_time = time.time()
    
    # Initialize tracking variables
    obs, _ = env.reset(seed=args.seed)
    obs_rms.update(np.array([obs]))  # Initialize normalization with first observation
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    # Print start message
    print(f"Starting training on Enhanced Maize environment with Vanilla DQN")
    print(f"Years range: {args.year1}-{args.year2}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Main training loop
    for global_step in range(args.total_timesteps):
        # Normalize observation
        normalized_obs = obs_rms.normalize(obs)
        
        # Epsilon-greedy action selection
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(torch.FloatTensor(normalized_obs).to(device))
            action = torch.argmax(q_values).cpu().numpy()
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Update observation normalization statistics
        obs_rms.update(np.array([next_obs]))
        
        # Process reward for better stability
        processed_reward = process_reward(reward)
        
        # Track original reward for stats
        episode_reward += reward
        episode_length += 1
        
        # Add experience to buffer
        rb.add(normalized_obs, obs_rms.normalize(next_obs), action, processed_reward, terminated)
        
        # Update observation
        obs = next_obs
        
        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                # Sample batch 
                data = rb.sample(args.batch_size)
                
                # Calculate TD targets
                with torch.no_grad():
                    # Vanilla DQN: target network does both selection and evaluation
                    next_q_values = target_network(data.next_observations).max(1)[0]

                    
                    # Calculate target values
                    td_target = data.rewards.flatten() + args.gamma * next_q_values * (1 - data.dones.flatten())
                
                # Calculate current Q-values
                current_q_values = q_network(data.observations).gather(1, data.actions).squeeze(1)
                
                # Calculate loss 
                td_errors = td_target - current_q_values
                loss = F.mse_loss(current_q_values, td_target)
                    
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
                optimizer.step()
                
                # Log metrics
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", current_q_values.mean().item(), global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    
                    
                    # Save for plotting
                    stats.add_training_info(loss.item(), current_q_values.mean().item())
            
            # Soft update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        
        # Handle episode completion
        if terminated or truncated:
            # Get episode info
            dry_yield = info.get("dry_yield", 0.0)
            total_irrigation = info.get("total_irrigation", 0.0)
            water_use_efficiency = info.get("water_use_efficiency", 0.0)
            
            # Record episode stats
            stats.add_episode(episode_reward, episode_length, dry_yield, total_irrigation, water_use_efficiency)
            
            # Log to tensorboard
            writer.add_scalar("charts/episodic_return", episode_reward, episode_count)
            writer.add_scalar("charts/episodic_length", episode_length, episode_count)
            writer.add_scalar("charts/dry_yield", dry_yield, episode_count)
            writer.add_scalar("charts/total_irrigation", total_irrigation, episode_count)
            if water_use_efficiency > 0:
                writer.add_scalar("charts/water_use_efficiency", water_use_efficiency, episode_count)
            
            # Get and log recent performance statistics
            (reward_mean, reward_std), (yield_mean, yield_std), (irr_mean, irr_std) = stats.get_recent_stats()
            if len(stats.recent_rewards) >= 10:  # Only log when we have enough episodes
                writer.add_scalar("charts/reward_mean_recent", reward_mean, episode_count)
                writer.add_scalar("charts/yield_mean_recent", yield_mean, episode_count)
                writer.add_scalar("charts/irrigation_mean_recent", irr_mean, episode_count)
            
            # Print progress with more informative stats
            print(f"Episode {episode_count} completed after {episode_length} steps")
            print(f"  Reward: {episode_reward:.2f}, Yield: {dry_yield:.2f}, Irrigation: {total_irrigation:.2f}")
            if water_use_efficiency > 0:
                print(f"  Water Use Efficiency: {water_use_efficiency:.4f}")
            
            if len(stats.recent_rewards) >= 10:
                print(f"  Recent performance (last 100 episodes):")
                print(f"    Reward: {reward_mean:.2f} ± {reward_std:.2f}")
                print(f"    Yield: {yield_mean:.2f} ± {yield_std:.2f}")
                print(f"    Irrigation: {irr_mean:.2f} ± {irr_std:.2f}")
            
            print(f"  Step: {global_step}/{args.total_timesteps}, SPS: {int(global_step / (time.time() - start_time))}")
            
            # Save intermediate model every 100 episodes
            if episode_count % 100 == 0 and args.save_model:
                checkpoint_dir = os.path.join(output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_ep{episode_count}.pt")
                torch.save({
                    'q_network_state_dict': q_network.state_dict(),
                    'target_network_state_dict': target_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode_count,
                    'global_step': global_step,
                    'obs_rms_mean': obs_rms.mean,
                    'obs_rms_var': obs_rms.var,
                }, checkpoint_path)
                print(f"  Checkpoint saved to {checkpoint_path}")
            
            # Reset for next episode
            obs, _ = env.reset()
            obs_rms.update(np.array([obs]))  # Update normalization with new observation
            episode_reward = 0
            episode_length = 0
            episode_count += 1
    
    # Training finished - plot stats
    stats.plot_stats()
    stats.plot_training_metrics()
    
    # Save final model
    if args.save_model:
        # Save full model with all needed components for inference
        model_path = os.path.join(output_dir, "final_model.pt")
        torch.save({
            'q_network_state_dict': q_network.state_dict(),
            'obs_rms_mean': obs_rms.mean,
            'obs_rms_var': obs_rms.var,
            'obs_space_shape': env.observation_space.shape,
            'action_space_n': env.action_space.n,
        }, model_path)
        print(f"Final model saved to {model_path}")
        
        # Also save a simple version of just the Q-network for easier loading
        simple_model_path = os.path.join(output_dir, "q_network.pt")
        torch.save(q_network.state_dict(), simple_model_path)
        print(f"Q-network weights saved to {simple_model_path}")
    
    # Record final training metrics
    final_time = time.time() - start_time
    final_sps = args.total_timesteps / final_time
    
    print(f"\nTraining completed:")
    print(f"  Total episodes: {episode_count}")
    print(f"  Total time: {final_time:.2f} seconds")
    print(f"  Average steps per second: {final_sps:.2f}")
    
    # Record final metrics for the experiment
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
    env.close()
    writer.close()
