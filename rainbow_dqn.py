# optimized_rainbow_dqn.py
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
from collections import deque
import logging
import warnings

# Import the EnhancedMaize environment
from aquacropgymnasium.env import EnhancedMaize

# Set up logging and warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up prettier plots (compatible with older versions)
try:
    plt.style.use('seaborn-whitegrid')  # For older versions
except:
    try:
        plt.style.use('ggplot')  # Alternative style
    except:
        pass  # Fall back to default style
sns.set_context("notebook", font_scale=1.2)

@dataclass
class Args:
    exp_name: str = "rainbow_dqn_enhanced_maize"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "enhanced_maize_rainbow_dqn"
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
    """the target network update rate"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
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
    
    # Rainbow specific parameters
    n_atoms: int = 101
    """number of atoms for distributional RL"""
    v_min: float = -100.0
    """minimum value of support"""
    v_max: float = 2000.0
    """maximum value of support"""
    n_step: int = 5 #3
    """number of steps for multi-step learning"""
    alpha: float = 0.6 #0.5
    """alpha parameter for prioritized replay"""
    beta: float = 0.4
    """initial beta parameter for prioritized replay"""
    beta_annealing: float = 0.0002
    """beta annealing rate"""
    
    # Custom Maize environment arguments
    year1: int = 1982
    """starting year for training (inclusive)"""
    year2: int = 2007
    """ending year for training (inclusive)"""


# Noisy Linear Layer for exploration
# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, std_init=0.5):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.std_init = std_init
        
#         # Learnable parameters
#         self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
#         self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
#         self.bias_mu = nn.Parameter(torch.empty(out_features))
#         self.bias_sigma = nn.Parameter(torch.empty(out_features))
#         self.register_buffer('bias_epsilon', torch.empty(out_features))
        
#         self.reset_parameters()
#         self.reset_noise()
    
#     def reset_parameters(self):
#         mu_range = 1 / np.sqrt(self.in_features)
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
#     def _scale_noise(self, size):
#         x = torch.randn(size)
#         return x.sign().mul(x.abs().sqrt())
    
#     def reset_noise(self):
#         epsilon_in = self._scale_noise(self.in_features)
#         epsilon_out = self._scale_noise(self.out_features)
#         self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
#         self.bias_epsilon.copy_(epsilon_out)
    
#     def forward(self, x):
#         if self.training:
#             weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
#             bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
#         else:
#             weight = self.weight_mu
#             bias = self.bias_mu
#         return F.linear(x, weight, bias)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):  # Lower std_init
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features).to(device))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features).to(device))
        
        # Factorized noise parameters
        self.register_buffer('epsilon_in', torch.empty(in_features).to(device))
        self.register_buffer('epsilon_out', torch.empty(out_features).to(device))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    # def reset_noise(self):
    #     self.epsilon_in = self._scale_noise(self.in_features)
    #     self.epsilon_out = self._scale_noise(self.out_features)
    #     # Use factorized noise for better exploration
    #     self.weight_epsilon = torch.outer(self.epsilon_out, self.epsilon_in)
    #     self.bias_epsilon = self.epsilon_out
    
    def reset_noise(self):
        device = self.weight_mu.device 
        self.epsilon_in = self._scale_noise(self.in_features).to(device)
        self.epsilon_out = self._scale_noise(self.out_features).to(device)
        # Use factorized noise for better exploration
        self.weight_epsilon = torch.outer(self.epsilon_out, self.epsilon_in)
        self.bias_epsilon = self.epsilon_out
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    
# Rainbow DQN Network with distributional RL and dueling architecture
# class RainbowDQN(nn.Module):
#     def __init__(self, observation_shape, action_space_n, n_atoms=51, v_min=-300, v_max=4000):
#         super().__init__()
#         self.action_space_n = action_space_n
#         self.n_atoms = n_atoms
#         self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
#         input_dim = int(np.prod(observation_shape))
        
#         # Feature extractor
#         self.features = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )
        
#         # Value stream (dueling architecture)
#         self.value_stream = nn.Sequential(
#             NoisyLinear(256, 128),
#             nn.ReLU(),
#             NoisyLinear(128, n_atoms)
#         )
        
#         # Advantage stream (dueling architecture)
#         self.advantage_stream = nn.Sequential(
#             NoisyLinear(256, 128),
#             nn.ReLU(),
#             NoisyLinear(128, action_space_n * n_atoms)
#         )
    
#     def forward(self, x):
#         batch_size = x.size(0)
#         features = self.features(x)
        
#         # Value and advantage streams
#         value_dist = self.value_stream(features).view(batch_size, 1, self.n_atoms)
#         advantage_dist = self.advantage_stream(features).view(batch_size, self.action_space_n, self.n_atoms)
        
#         # Combine using dueling formula
#         q_dist = value_dist + advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        
#         # Convert to probabilities
#         return F.softmax(q_dist, dim=2)
    
#     def reset_noise(self):
#         """Reset noise for all noisy layers"""
#         for module in self.modules():
#             if isinstance(module, NoisyLinear):
#                 module.reset_noise()
    
#     def act(self, state, device):
#         """Select action greedily"""
#         with torch.no_grad():
#             state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#             dist = self.forward(state)
#             expected_value = torch.sum(dist * self.support, dim=2)
#             action = expected_value.argmax(1).item()
#         return action

class RainbowDQN(nn.Module):
    def __init__(self, observation_shape, action_space_n, n_atoms=101, v_min=-100, v_max=2000):
        super().__init__()
        self.action_space_n = action_space_n
        self.n_atoms = n_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        
        input_dim = int(np.prod(observation_shape))
        
        # Enhanced feature extractor with layer normalization and residual connections
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Value stream with residual connection
        self.value_hidden = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        self.value_output = NoisyLinear(128, n_atoms)
        
        # Advantage stream with residual connection
        self.advantage_hidden = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        self.advantage_output = NoisyLinear(128, action_space_n * n_atoms)
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.features(x)
        
        # Process value and advantage streams
        value_hidden = self.value_hidden(features)
        value_dist = self.value_output(value_hidden).view(batch_size, 1, self.n_atoms)
        
        advantage_hidden = self.advantage_hidden(features)
        advantage_dist = self.advantage_output(advantage_hidden).view(batch_size, self.action_space_n, self.n_atoms)
        
        # Combine using dueling formula
        q_dist = value_dist + advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        
        # Convert to probabilities
        return F.softmax(q_dist, dim=2)
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
    
    def act(self, state, device):
        """Select action greedily"""
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            dist = self.forward(state)
            expected_value = torch.sum(dist * self.support, dim=2)
            action = expected_value.argmax(1).item()
        return action
# Prioritized Experience Replay Buffer with N-step learning
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, observation_shape, n_actions, device, 
                 n_step=3, gamma=0.99, alpha=0.5, beta=0.6, beta_annealing=0.001):
        self.buffer_size = buffer_size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        # PER parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = 1e-6  # small value to avoid zero priority
        
        # Create buffers
        self.observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Priority storage
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        
        # N-step return buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        self.pos = 0
        self.full = False
        self.max_priority = 1.0
    
    def _get_n_step_info(self):
        """Returns rewards and next observation after n steps"""
        reward, next_obs, done = self.n_step_buffer[-1][-3:]
        
        # Calculate n-step return
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            
            reward = r + self.gamma * reward * (1 - d)
            done = done or d
            next_obs = n_o if d else next_obs
            
        return reward, next_obs, done
    
    def add(self, obs, action, reward, next_obs, done):
        # Store transition in n-step buffer
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        
        # If we don't have enough transitions for n-step learning, return
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Get n-step return
        reward, next_obs, done = self._get_n_step_info()
        obs = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        
        # Store in replay buffer
        self.observations[self.pos] = np.array(obs, dtype=np.float32)
        self.next_observations[self.pos] = np.array(next_obs, dtype=np.float32)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
        # Set priority to maximum priority initially
        self.priorities[self.pos] = self.max_priority ** self.alpha
        
        # Update buffer position
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def sample(self, batch_size):
        """Sample a batch of experiences with prioritization"""
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        # Calculate sampling range
        upper_bound = self.buffer_size if self.full else self.pos
        
        if upper_bound == 0:
            return None  # Buffer is empty
            
        # Get priorities and calculate sampling probabilities
        priorities = self.priorities[:upper_bound]
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(upper_bound, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (upper_bound * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Create batch tensors
        observations = torch.as_tensor(self.observations[indices], dtype=torch.float32).to(self.device)
        next_observations = torch.as_tensor(self.next_observations[indices], dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.long).to(self.device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32).to(self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32).to(self.device)
        
        # Return batch info
        return observations, actions, next_observations, dones, rewards, weights, indices
    
    # def update_priorities(self, indices, priorities):
    #     """Update priorities of sampled transitions"""
    #     priorities = np.abs(priorities) + self.epsilon
    #     self.priorities[indices] = priorities ** self.alpha
    #     self.max_priority = max(self.max_priority, priorities.max())
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions with clipping"""
        priorities = np.clip(np.abs(priorities) + self.epsilon, 1e-6, 100.0)  # Clip to reasonable range
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = min(100.0, max(self.max_priority, priorities.max()))  # Cap max priority


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


def projection_distribution(next_dist, next_action, reward, done, support, n_atoms, gamma):
    """Project next_distribution into current distribution for Categorical DQN"""
    batch_size = next_dist.size(0)
    delta_z = float(support[1] - support[0])
    support = support.expand_as(next_dist)
    
    # Compute projected distribution
    reward = reward.unsqueeze(1).expand(batch_size, n_atoms)
    done = done.unsqueeze(1).expand(batch_size, n_atoms)
    support = reward + (1 - done) * gamma * support
    
    # Clamp support to ensure it's within bounds
    support = support.clamp(min=support[0, 0], max=support[0, -1])
    
    # Compute projection
    b = (support - support[0, 0]) / delta_z
    lower_bound = b.floor().long()
    upper_bound = b.ceil().long()
    
    # Handle corner cases
    lower_bound[(upper_bound > 0) * (lower_bound == upper_bound)] -= 1
    upper_bound[(lower_bound < (n_atoms - 1)) * (lower_bound == upper_bound)] += 1
    
    # Project
    proj_dist = torch.zeros_like(next_dist)
    offset = torch.linspace(0, (batch_size - 1) * n_atoms, batch_size).long().unsqueeze(1).expand(batch_size, n_atoms).to(next_dist.device)
    
    # Distribute probability mass
    proj_dist.view(-1).index_add_(
        0, 
        (lower_bound + offset).view(-1), 
        (next_dist * (upper_bound.float() - b)).view(-1)
    )
    proj_dist.view(-1).index_add_(
        0, 
        (upper_bound + offset).view(-1), 
        (next_dist * (b - lower_bound.float())).view(-1)
    )
    
    return proj_dist


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear interpolation schedule for epsilon"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_env(year1, year2, seed):
    """Create a wrapped, monitored Enhanced Maize environment"""
    def thunk():
        env = EnhancedMaize(mode='train', year1=year1, year2=year2)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return thunk


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


# def process_reward(reward):
#     """Process raw rewards for better stability"""
#     # Use tanh to bound rewards without clipping information
#     return np.tanh(reward / 1000.0)
    
def process_reward(reward, episode_step=0, is_terminal=False):
    """
    Enhanced reward processing with scale adjustment and terminal bonus
    """
    # Basic stabilization with tanh
    processed = np.tanh(reward / 1000.0)
    
    # Terminal state gets higher weight
    if is_terminal:
        processed *= 1.5
    
    # Early steps get lower weight to focus on later consequences
    if episode_step < 30:  # Assuming early growth stage
        processed *= 0.8
    
    return processed

def cosine_annealing_lr(initial_lr, step, total_steps, min_lr=1e-5):
    """Cosine annealing learning rate schedule"""
    if step >= total_steps:
        return min_lr
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(step / total_steps * np.pi))



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
    
    # Create statistics tracker
    stats = EpisodeStats(output_dir)
    
    # Initialize observation normalization
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    
    # Create Rainbow DQN network
    online_net = RainbowDQN(
        env.observation_space.shape, 
        env.action_space.n,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max
    ).to(device)
    
    target_net = RainbowDQN(
        env.observation_space.shape, 
        env.action_space.n,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max
    ).to(device)
    
    target_net.load_state_dict(online_net.state_dict())
    
    # Create optimizer
    optimizer = optim.Adam(online_net.parameters(), lr=args.learning_rate)
    
    # Create prioritized replay buffer with n-step returns
    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        env.observation_space.shape,
        env.action_space.n,
        device,
        n_step=args.n_step,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        beta_annealing=args.beta_annealing
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
    print(f"Starting training on Enhanced Maize environment with Rainbow DQN")
    print(f"Years range: {args.year1}-{args.year2}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Support tensor for distributional RL
    support = torch.linspace(args.v_min, args.v_max, args.n_atoms).to(device)

    
    # Main training loop
    for global_step in range(args.total_timesteps):
        # Reset noise periodically
        # if global_step % 1000 == 0:
        #     online_net.reset_noise()
        #     target_net.reset_noise()
        noise_reset_freq = int(1000 * (1 + 2 * global_step / args.total_timesteps))
        noise_reset_freq = min(noise_reset_freq, 5000)  # Cap at reasonable value

        if global_step % noise_reset_freq == 0:
            online_net.reset_noise()
            # Only reset target net noise less frequently
            if global_step % (noise_reset_freq * 3) == 0:
                target_net.reset_noise()
        
        # Normalize observation
        normalized_obs = obs_rms.normalize(obs)

        action = online_net.act(normalized_obs, device)

        
        # Epsilon-greedy action selection (only for early training)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            # Use expected values of distribution to select action
            # with torch.no_grad():
            #     q_dist = online_net(torch.FloatTensor(normalized_obs).to(device).unsqueeze(0))
            #     q_values = (q_dist * support).sum(dim=2)
            #     action = q_values.argmax(1).item()
            action = online_net.act(normalized_obs, device)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Update observation normalization statistics
        obs_rms.update(np.array([next_obs]))
        
        # Process reward for better stability
        processed_reward = process_reward(reward, episode_length, terminated)
        
        # Track original reward for stats
        episode_reward += reward
        episode_length += 1
        
        # Add experience to buffer
        rb.add(normalized_obs, action, processed_reward, obs_rms.normalize(next_obs), terminated or truncated)
        
        # Update observation
        obs = next_obs
        
        # Training
        if global_step > args.learning_starts:
            train_freq = max(1, int(args.train_frequency * (1.0 - global_step / args.total_timesteps)))
            if global_step % train_freq == 0:
            # if global_step % args.train_frequency == 0:
                # Sample batch with priorities
                batch = rb.sample(args.batch_size)
                
                if batch is not None:
                    observations, actions, next_observations, dones, rewards, weights, indices = batch
                    
                    # Get current state distribution
                    current_dist = online_net(observations)
                    current_dist = current_dist[range(args.batch_size), actions]
                    
                    # Compute greedy actions from online network for next states (Double DQN)
                    with torch.no_grad():
                        # Select actions using online network
                        next_dist = online_net(next_observations)
                        next_q = (next_dist * support).sum(2)
                        next_actions = next_q.argmax(1)
                        
                        # Evaluate actions using target network
                        next_dist = target_net(next_observations)
                        next_dist = next_dist[range(args.batch_size), next_actions]
                        
                        # Project next distribution (Distributional RL)
                        target_dist = projection_distribution(
                            next_dist, 
                            next_actions, 
                            rewards, 
                            dones, 
                            support, 
                            args.n_atoms, 
                            args.gamma ** args.n_step
                        )
                    
                    # Compute categorical cross-entropy loss
                    log_probs = torch.log(current_dist + 1e-5)  # Add small constant for numerical stability
                    elementwise_loss = -(target_dist * log_probs).sum(1)
                    
                    # Apply importance sampling weights from PER
                    loss = (elementwise_loss * weights).mean()

                    # Add this inside the training loop, before optimizer.zero_grad()
                current_lr = cosine_annealing_lr(args.learning_rate, global_step, args.total_timesteps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                    
                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                    optimizer.step()
                    
                    # Update priorities in buffer
                    with torch.no_grad():
                        # TD errors for PER update (use expected values)
                        current_q = (current_dist * support).sum(1).detach()
                        target_q = (target_dist * support).sum(1).detach()
                        td_error = torch.abs(current_q - target_q).cpu().numpy()
                        rb.update_priorities(indices, td_error)
                    
                    # Calculate mean Q-value for logging
                    q_value = (current_dist * support).mean().item()
                    
                    # Log metrics
                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss.item(), global_step)
                        writer.add_scalar("losses/q_values", q_value, global_step)
                        writer.add_scalar("charts/epsilon", epsilon, global_step)
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                        
                        # Save for plotting
                        stats.add_training_info(loss.item(), q_value)
                        
            # Update target network
            # if global_step % args.target_network_frequency == 0:
            #     for target_param, param in zip(target_net.parameters(), online_net.parameters()):
            #         target_param.data.copy_(
            #             args.tau * param.data + (1.0 - args.tau) * target_param.data
            #         )
            if global_step % args.target_network_frequency == 0:
                # Adaptive tau based on training progress
                adaptive_tau = args.tau * (1.0 + global_step / args.total_timesteps)
                adaptive_tau = min(0.1, adaptive_tau)  # Cap at reasonable value
                
                for target_param, param in zip(target_net.parameters(), online_net.parameters()):
                    target_param.data.copy_(
                        adaptive_tau * param.data + (1.0 - adaptive_tau) * target_param.data
                    )
        
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

            # Add early stopping based on recent performance
            if episode_count > 200 and episode_count % 50 == 0:
                recent_performance = np.mean(stats.recent_rewards)
                
                # Save best model based on performance
                if not hasattr(stats, 'best_performance') or recent_performance > stats.best_performance:
                    stats.best_performance = recent_performance
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'online_net_state_dict': online_net.state_dict(),
                        'obs_rms_mean': obs_rms.mean,
                        'obs_rms_var': obs_rms.var,
                        'obs_space_shape': env.observation_space.shape,
                        'action_space_n': env.action_space.n,
                        'n_atoms': args.n_atoms,
                        'v_min': args.v_min,
                        'v_max': args.v_max,
                        'performance': recent_performance
                    }, best_model_path)
                    print(f"  New best model saved with performance: {recent_performance:.2f}")
            
            # Save intermediate model every 100 episodes
            if episode_count % 100 == 0 and args.save_model:
                checkpoint_dir = os.path.join(output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_ep{episode_count}.pt")
                torch.save({
                    'online_net_state_dict': online_net.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
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
            'online_net_state_dict': online_net.state_dict(),
            'obs_rms_mean': obs_rms.mean,
            'obs_rms_var': obs_rms.var,
            'obs_space_shape': env.observation_space.shape,
            'action_space_n': env.action_space.n,
            'n_atoms': args.n_atoms,
            'v_min': args.v_min,
            'v_max': args.v_max,
        }, model_path)
        print(f"Final model saved to {model_path}")
        
        # Also save a simple version of just the online network for easier loading
        simple_model_path = os.path.join(output_dir, "rainbow_net.pt")
        torch.save(online_net.state_dict(), simple_model_path)
        print(f"Rainbow network weights saved to {simple_model_path}")
    
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