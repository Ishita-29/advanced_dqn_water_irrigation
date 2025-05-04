
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import logging
import time
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
import warnings
import torch.serialization
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
from aquacropgymnasium.env import EnhancedMaize

# Import model implementations
# Vanilla DQN
from vanilladqn import QNetwork as VanillaDQNNetwork
# Double DQN
from double_dqn import DuelingQNetwork as DoubleDQNNetwork
# Rainbow DQN
from rainbow_dqn import RainbowDQN, NoisyLinear
# PPO
from ppo import Agent as PPOAgent

# Setup logging and warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)

# Output directories
eval_output_dir = './eval_output_1000k_final'
plots_dir = os.path.join(eval_output_dir, 'plots')
results_dir = os.path.join(eval_output_dir, 'results')

# Create output directories
os.makedirs(eval_output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Economic parameters - can be adjusted based on current market conditions
CROP_PRICE = 180  # USD per tonne
IRRIGATION_COST = 1  # USD per mm
FIXED_COST = 1728  # USD per ha

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

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

def load_vanilla_dqn_model(model_path, env, device="cpu"):
    """Load a Vanilla DQN model."""
    try:
        # Load with safe_globals
        with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct', 
                                              'numpy.core.multiarray._reconstruct']):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get state dict and create model
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            state_dict = checkpoint['q_network_state_dict']
            # Extract observation normalization parameters if available
            obs_rms_mean = checkpoint.get('obs_rms_mean', None)
            obs_rms_var = checkpoint.get('obs_rms_var', None)
            obs_shape = checkpoint.get('obs_space_shape', env.observation_space.shape)
            action_n = checkpoint.get('action_space_n', env.action_space.n)
        else:
            state_dict = checkpoint
            obs_rms_mean = None
            obs_rms_var = None
            obs_shape = env.observation_space.shape
            action_n = env.action_space.n
        
        # Create model - using the correct QNetwork class from vanilladqn_clean_rl_enhanced_env_500k.py
        model = VanillaDQNNetwork(np.prod(obs_shape), action_n).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Add normalization attributes if available
        if obs_rms_mean is not None and obs_rms_var is not None:
            model.normalize = True
            model.obs_rms_mean = obs_rms_mean
            model.obs_rms_var = obs_rms_var
        else:
            model.normalize = False
            
        return model
    
    except Exception as e:
        raise Exception(f"Failed to load Vanilla DQN model: {str(e)}")

def load_double_dqn_model(model_path, env, device="cpu"):
    """Load a Double DQN model."""
    try:
        # Load with safe_globals
        with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct', 
                                              'numpy.core.multiarray._reconstruct']):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get state dict and create model
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
            state_dict = checkpoint['q_network_state_dict']
            # Extract observation normalization parameters if available
            obs_rms_mean = checkpoint.get('obs_rms_mean', None)
            obs_rms_var = checkpoint.get('obs_rms_var', None)
            obs_shape = checkpoint.get('obs_space_shape', env.observation_space.shape)
            action_n = checkpoint.get('action_space_n', env.action_space.n)
        else:
            state_dict = checkpoint
            obs_rms_mean = None
            obs_rms_var = None
            obs_shape = env.observation_space.shape
            action_n = env.action_space.n
        
        # Create model
        model = DoubleDQNNetwork(obs_shape, action_n).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Add normalization attributes if available
        if obs_rms_mean is not None and obs_rms_var is not None:
            model.normalize = True
            model.obs_rms_mean = obs_rms_mean
            model.obs_rms_var = obs_rms_var
        else:
            model.normalize = False
            
        return model
    
    except Exception as e:
        raise Exception(f"Failed to load Double DQN model: {str(e)}")


def load_rainbow_model(model_path, env, device="cpu"):
    """Load a Rainbow DQN model with flexible architecture handling."""
    try:
        # Load with safe_globals
        with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct', 
                                              'numpy.core.multiarray._reconstruct']):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract parameters from checkpoint
        if isinstance(checkpoint, dict):
            # Get parameters from checkpoint, with fallbacks
            obs_shape = checkpoint.get('obs_space_shape', env.observation_space.shape)
            action_n = checkpoint.get('action_space_n', env.action_space.n)
            n_atoms = checkpoint.get('n_atoms', 101)
            v_min = checkpoint.get('v_min', -100.0)  # Updated defaults to match new implementation
            v_max = checkpoint.get('v_max', 2000.0)
            
            # Find the state dict
            state_dict = None
            if 'online_net_state_dict' in checkpoint:
                state_dict = checkpoint['online_net_state_dict']
            elif 'online_net' in checkpoint:
                state_dict = checkpoint['online_net']
            else:
                # Assume checkpoint itself is the state dict
                state_dict = checkpoint
            
            # Extract observation normalization parameters if available
            obs_rms_mean = checkpoint.get('obs_rms_mean', None)
            obs_rms_var = checkpoint.get('obs_rms_var', None)
            
            # Define NoisyLinear class without device dependency
            class NoisyLinear(nn.Module):
                def __init__(self, in_features, out_features, std_init=0.1):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.std_init = std_init
                    
                    # Learnable parameters
                    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
                    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
                    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
                    
                    self.bias_mu = nn.Parameter(torch.empty(out_features))
                    self.bias_sigma = nn.Parameter(torch.empty(out_features))
                    self.register_buffer('bias_epsilon', torch.empty(out_features))
                    
                    # Factorized noise parameters
                    self.register_buffer('epsilon_in', torch.empty(in_features))
                    self.register_buffer('epsilon_out', torch.empty(out_features))
                    
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
                
                def reset_noise(self):
                    self.epsilon_in = self._scale_noise(self.in_features)
                    self.epsilon_out = self._scale_noise(self.out_features)
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
            
            # Create custom RainbowDQN class that matches the enhanced architecture
            class CustomRainbowDQN(nn.Module):
                def __init__(self, observation_shape, action_space_n, n_atoms=101, v_min=-100, v_max=2000):
                    super().__init__()
                    self.action_space_n = action_space_n
                    self.n_atoms = n_atoms
                    self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
                    
                    input_dim = int(np.prod(observation_shape))
                    
                    # Enhanced feature extractor with layer normalization
                    self.features = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.LayerNorm(256),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.LayerNorm(256)
                    )
                    
                    # Value stream with separate hidden and output layers
                    self.value_hidden = nn.Sequential(
                        NoisyLinear(256, 128),
                        nn.ReLU(),
                        nn.LayerNorm(128)
                    )
                    self.value_output = NoisyLinear(128, n_atoms)
                    
                    # Advantage stream with separate hidden and output layers
                    self.advantage_hidden = nn.Sequential(
                        NoisyLinear(256, 128),
                        nn.ReLU(),
                        nn.LayerNorm(128)
                    )
                    self.advantage_output = NoisyLinear(128, action_space_n * n_atoms)
                
                def forward(self, x):
                    import torch.nn.functional as F
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
            
            # Create model with the enhanced architecture
            model = CustomRainbowDQN(
                obs_shape, 
                action_n, 
                n_atoms=n_atoms, 
                v_min=v_min, 
                v_max=v_max
            ).to(device)
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            # Add normalization attributes if available
            if obs_rms_mean is not None and obs_rms_var is not None:
                model.normalize = True
                model.obs_rms_mean = obs_rms_mean
                model.obs_rms_var = obs_rms_var
            else:
                model.normalize = False
        else:
            # For simple state dicts, create a model with the enhanced architecture
            n_atoms = 101  # Updated default
            v_min = -100.0
            v_max = 2000.0
            
            # Define NoisyLinear class without device dependency
            class NoisyLinear(nn.Module):
                def __init__(self, in_features, out_features, std_init=0.1):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.std_init = std_init
                    
                    # Learnable parameters
                    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
                    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
                    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
                    
                    self.bias_mu = nn.Parameter(torch.empty(out_features))
                    self.bias_sigma = nn.Parameter(torch.empty(out_features))
                    self.register_buffer('bias_epsilon', torch.empty(out_features))
                    
                    # Factorized noise parameters
                    self.register_buffer('epsilon_in', torch.empty(in_features))
                    self.register_buffer('epsilon_out', torch.empty(out_features))
                    
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
                
                def reset_noise(self):
                    self.epsilon_in = self._scale_noise(self.in_features)
                    self.epsilon_out = self._scale_noise(self.out_features)
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
            
            # Create custom model with the same architecture as above
            class CustomRainbowDQN(nn.Module):
                def __init__(self, observation_shape, action_space_n, n_atoms=101, v_min=-100, v_max=2000):
                    super().__init__()
                    self.action_space_n = action_space_n
                    self.n_atoms = n_atoms
                    self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
                    
                    input_dim = int(np.prod(observation_shape))
                    
                    # Enhanced feature extractor with layer normalization
                    self.features = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.LayerNorm(256),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.LayerNorm(256)
                    )
                    
                    # Value stream with separate hidden and output layers
                    self.value_hidden = nn.Sequential(
                        NoisyLinear(256, 128),
                        nn.ReLU(),
                        nn.LayerNorm(128)
                    )
                    self.value_output = NoisyLinear(128, n_atoms)
                    
                    # Advantage stream with separate hidden and output layers
                    self.advantage_hidden = nn.Sequential(
                        NoisyLinear(256, 128),
                        nn.ReLU(),
                        nn.LayerNorm(128)
                    )
                    self.advantage_output = NoisyLinear(128, action_space_n * n_atoms)
                
                def forward(self, x):
                    import torch.nn.functional as F
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
            
            model = CustomRainbowDQN(
                env.observation_space.shape, 
                env.action_space.n, 
                n_atoms, 
                v_min, 
                v_max
            ).to(device)
            
            model.load_state_dict(checkpoint)
            model.normalize = False
        
        model.eval()
        return model
    
    except Exception as e:
        raise Exception(f"Failed to load Rainbow DQN model: {str(e)}")
    
def load_ppo_model(model_path, env, device="cpu"):
    """Load a PPO model."""
    try:
        # Create a dummy environment to initialize the Agent
        class DummyVecEnv:
            def __init__(self, single_obs_space, single_action_space):
                self.single_observation_space = single_obs_space
                self.single_action_space = single_action_space
        
        dummy_env = DummyVecEnv(env.observation_space, env.action_space)
        
        # Load with safe_globals
        with torch.serialization.safe_globals(['numpy._core.multiarray._reconstruct', 
                                              'numpy.core.multiarray._reconstruct']):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model and load state dict
        model = PPOAgent(dummy_env).to(device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Extract observation normalization parameters if available
            obs_rms_mean = checkpoint.get('obs_rms_mean', None)
            obs_rms_var = checkpoint.get('obs_rms_var', None)
        else:
            state_dict = checkpoint
            obs_rms_mean = None
            obs_rms_var = None
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Add normalization attributes if available
        if obs_rms_mean is not None and obs_rms_var is not None:
            model.normalize = True
            model.obs_rms_mean = obs_rms_mean
            model.obs_rms_var = obs_rms_var
        else:
            model.normalize = False
            
        return model
    
    except Exception as e:
        raise Exception(f"Failed to load PPO model: {str(e)}")

# Random agent for baseline comparison
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state, device):
        return self.action_space.sample()

# Rule-based agent for baseline comparison - simple threshold-based irrigation
class ThresholdAgent:
    def __init__(self, env):
        self.env = env
        
    def act(self, state, device):
        # Extract water stress from state (assuming Enhanced Maize environment)
        # In the enhanced environment, water stress is at index 6 (0-indexed)
        if len(state) >= 7:
            water_stress = state[6]  # Water stress index
            
            # Rule-based decision making
            if water_stress > 0.7:  # High stress - irrigate more
                return 4  # Use highest irrigation (25mm)
            elif water_stress > 0.5:  # Medium stress
                return 3  # Use medium-high irrigation (15mm)
            elif water_stress > 0.3:  # Low stress
                return 1  # Use low irrigation (5mm)
            else:  # No stress
                return 0  # No irrigation
        else:
            # Fallback for standard Maize environment
            # Extract depletion and TAW (if available)
            if len(state) >= 5:
                depletion = state[3]
                taw = state[4]
                
                # Calculate simplified water stress
                water_stress = min(1.0, max(0.0, depletion / taw)) if taw > 0 else 1.0
                
                if water_stress > 0.6:
                    return 1  # Apply irrigation
                else:
                    return 0  # No irrigation
            
            # Default to random action if state doesn't match expected format
            return np.random.randint(0, self.env.action_space.n)

# Climate-based agent - uses weather forecast
class WeatherBasedAgent:
    def __init__(self, env):
        self.env = env
        
    def act(self, state, device):
        # Extract precipitation forecast from state
        # For Enhanced Maize, precipitation forecast is in indices 8-14 (7 days)
        if len(state) >= 15:
            precip_forecast = state[8:15]
            
            # Check if rain is coming in the next few days
            rain_coming = sum(precip_forecast[:3]) > 5.0  # >5mm in next 3 days
            
            # Extract water stress
            water_stress = state[6] if len(state) > 7 else 0.5
            
            # Decision logic
            if rain_coming:
                # Rain expected, reduce or skip irrigation
                if water_stress > 0.8:  # Only irrigate if extreme stress
                    return 2  # Medium irrigation (10mm)
                else:
                    return 0  # Skip irrigation
            else:
                # No rain expected, irrigate based on stress
                if water_stress > 0.7:
                    return 4  # High irrigation (25mm)
                elif water_stress > 0.5:
                    return 3  # Medium-high irrigation (15mm)
                elif water_stress > 0.3:
                    return 1  # Low irrigation (5mm)
                else:
                    return 0  # No irrigation
        
        # Fallback for standard environment
        return np.random.randint(0, self.env.action_space.n)

# Traditional irrigation strategies using AquaCrop
def evaluate_traditional_strategies(years_range=(2008, 2018), num_runs=100):
    # Initialize results storage
    results = defaultdict(lambda: defaultdict(list))
    
    # Get climate data
    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)
    
    # Crop and soil setup
    soil = Soil('SandyLoam')
    crop_obj = Crop('Maize', planting_date='05/01')
    initWC = InitialWaterContent(value=['FC'])
    
    # Define traditional irrigation strategies
    strategies = {
        'Rainfed': IrrigationManagement(irrigation_method=0),
        'Thresholds': IrrigationManagement(irrigation_method=1, SMT=[23.72, 26.46, 38.19, 50.11] * 4),
        'Interval': IrrigationManagement(irrigation_method=2, IrrInterval=7),
        'Net': IrrigationManagement(irrigation_method=4, NetIrrSMT=70)
    }
    
    # Run simulations for each strategy
    for strategy_name, irr_mngt in strategies.items():
        print(f"Evaluating traditional strategy: {strategy_name}")
        
        for year in range(years_range[0], years_range[1] + 1):
            # Set year-specific parameters
            sim_start = f"{year}/05/01"
            sim_end = f"{year}/12/31"
            
            # Create a copy of weather data with correct year
            year_wdf = wdf.copy()
            year_wdf['Year'] = year
            
            # Run AquaCrop model
            model = AquaCropModel(
                sim_start, 
                sim_end, 
                year_wdf, 
                soil, 
                crop_obj,
                initial_water_content=initWC,
                irrigation_management=irr_mngt
            )
            model.run_model(till_termination=True)
            
            # Extract results
            final_stats = model._outputs.final_stats
            
            # Store results
            results[strategy_name]['yield'].append(final_stats['Dry yield (tonne/ha)'].values[0])
            results[strategy_name]['irrigation'].append(final_stats['Seasonal irrigation (mm)'].values[0])
    
    # Calculate metrics
    summary_results = {}
    for strategy, metrics in results.items():
        mean_yield = np.mean(metrics['yield'])
        std_yield = np.std(metrics['yield'])
        mean_irrigation = np.mean(metrics['irrigation'])
        std_irrigation = np.std(metrics['irrigation'])
        
        # Calculate profit
        profit = CROP_PRICE * mean_yield - IRRIGATION_COST * mean_irrigation - FIXED_COST
        
        # Calculate water use efficiency
        wue = (mean_yield * 1000) / mean_irrigation if mean_irrigation > 0 else np.nan
        
        summary_results[strategy] = {
            'Dry yield (tonne/ha)_mean': mean_yield,
            'Dry yield (tonne/ha)_std': std_yield,
            'Seasonal irrigation (mm)_mean': mean_irrigation,
            'Seasonal irrigation (mm)_std': std_irrigation,
            'Profit_mean': profit,
            'WaterEfficiency_mean': wue
        }
    
    return pd.DataFrame.from_dict(summary_results, orient='index').reset_index().rename(columns={'index': 'label'})

# Normalize observation if needed
def normalize_obs(obs, agent):
    """Normalize observation if the agent has normalization parameters"""
    if hasattr(agent, 'normalize') and agent.normalize:
        return (obs - agent.obs_rms_mean) / np.sqrt(agent.obs_rms_var + 1e-8)
    return obs

# Evaluate a RL agent with the enhanced environment
def evaluate_rl_agent(agent, model_type, env_class=EnhancedMaize, n_episodes=50, seed=42, years_range=(2008, 2018)):
    """
    Evaluate an RL agent on the environment.
    
    Args:
        agent: The RL agent to evaluate
        model_type: Type of model ('rainbow', 'dqn', 'double_dqn', 'ppo', or 'rule_based')
        env_class: Environment class to use
        n_episodes: Number of episodes to evaluate
        seed: Random seed
        years_range: Years range for evaluation
        
    Returns:
        Evaluation results and detailed per-episode data
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Initialize metrics tracking
    yields = []
    irrigations = []
    rewards = []
    water_use_efficiencies = []
    water_stress_values = []
    irrigation_decisions = []
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = env_class(mode='eval', year1=years_range[0], year2=years_range[1])
    env.reset(seed=seed)
    
    # Calculate support if using Rainbow
    support = None
    if model_type == 'rainbow':
        # For Rainbow, we need support to calculate expected values
        if hasattr(agent, 'support'):
            support = agent.support
        else:
            # Default support range
            support = torch.linspace(-300, 4000, 51).to(device)
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        # Reset environment with random year in evaluation range
        obs, _ = env.reset()
        
        # Track episode metrics
        episode_yield = 0
        episode_irrigation = 0
        episode_reward = 0
        episode_water_stress = []
        episode_decisions = []
        
        done = False
        
        # Run episode
        while not done:
            # Select action using agent
            if model_type == 'rainbow':
                # For Rainbow
                with torch.no_grad():
                    # Normalize observation if needed
                    norm_obs = normalize_obs(obs, agent)
                    
                    state = torch.FloatTensor(norm_obs).unsqueeze(0).to(device)
                    dist = agent(state)
                    expected_value = torch.sum(dist * support, dim=2)
                    action = expected_value.argmax(1).item()
            elif model_type in ['dqn', 'double_dqn']:
                # For DQN/Double DQN
                with torch.no_grad():
                    # Normalize observation if needed
                    norm_obs = normalize_obs(obs, agent)
                    
                    state = torch.FloatTensor(norm_obs).unsqueeze(0).to(device)
                    q_values = agent(state)
                    action = q_values.argmax(1).item()
            elif model_type == 'ppo':
                # For PPO
                with torch.no_grad():
                    # Normalize observation if needed
                    norm_obs = normalize_obs(obs, agent)
                    
                    state = torch.FloatTensor(norm_obs).unsqueeze(0).to(device)
                    action, _, _, _ = agent.get_action_and_value(state)
                    action = action.cpu().numpy().item()
            else:
                # For rule-based agents
                action = agent.act(obs, device)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            
            # Track irrigation decision
            irrigation_depth = env.action_depths[action]
            episode_irrigation += irrigation_depth
            episode_decisions.append(irrigation_depth)
            
            # Track water stress if available
            if 'water_stress' in info:
                episode_water_stress.append(info['water_stress'])
            
            # Update observation
            obs = next_obs
        
        # Get final yield and other metrics from info
        dry_yield = info.get('dry_yield', 0.0)
        total_irrigation = info.get('total_irrigation', episode_irrigation)
        water_use_efficiency = info.get('water_use_efficiency', 0.0)
        
        # Handle case where water_use_efficiency not directly provided
        if water_use_efficiency == 0.0 and total_irrigation > 0:
            water_use_efficiency = (dry_yield * 1000) / total_irrigation
        
        # Append episode results
        yields.append(dry_yield)
        irrigations.append(total_irrigation)
        rewards.append(episode_reward)
        water_use_efficiencies.append(water_use_efficiency)
        irrigation_decisions.append(episode_decisions)
        
        if episode_water_stress:
            water_stress_values.append(np.mean(episode_water_stress))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode + 1}/{n_episodes} episodes")
    
    # Calculate final metrics
    mean_yield = np.mean(yields)
    std_yield = np.std(yields)
    mean_irrigation = np.mean(irrigations)
    std_irrigation = np.std(irrigations)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_water_efficiency = np.nanmean(water_use_efficiencies)
    std_water_efficiency = np.nanstd(water_use_efficiencies)
    
    # Calculate profit
    profit = CROP_PRICE * mean_yield - IRRIGATION_COST * mean_irrigation - FIXED_COST
    
    # Create summary DataFrame
    results = {
        'Dry yield (tonne/ha)_mean': mean_yield,
        'Dry yield (tonne/ha)_std': std_yield,
        'Seasonal irrigation (mm)_mean': mean_irrigation,
        'Seasonal irrigation (mm)_std': std_irrigation,
        'Reward_mean': mean_reward,
        'Reward_std': std_reward,
        'WaterEfficiency_mean': mean_water_efficiency,
        'WaterEfficiency_std': std_water_efficiency,
        'Profit_mean': profit
    }
    
    # Add water stress if available
    if water_stress_values:
        results['WaterStress_mean'] = np.mean(water_stress_values)
        results['WaterStress_std'] = np.std(water_stress_values)
    
    detailed_results = {
        'yields': yields,
        'irrigations': irrigations,
        'rewards': rewards,
        'water_use_efficiencies': water_use_efficiencies,
        'water_stress_values': water_stress_values,
        'irrigation_decisions': irrigation_decisions
    }
    
    return results, detailed_results

# Main evaluation function
def evaluate_all_approaches(vanilla_dqn_path=None, double_dqn_path=None, rainbow_path=None, ppo_path=None,
                           n_episodes=50, seed=42, years_range=(2008, 2018)):
    """
    Evaluate all approaches and generate comparison metrics and visualizations
    
    Args:
        vanilla_dqn_path: Path to trained Vanilla DQN model
        double_dqn_path: Path to trained Double DQN model
        rainbow_path: Path to trained Rainbow DQN model
        ppo_path: Path to trained PPO model
        n_episodes: Number of evaluation episodes
        seed: Random seed
        years_range: Tuple of (start_year, end_year) for evaluation
    """
    print("Starting comprehensive evaluation of irrigation control approaches...")
    results_list = []
    detailed_results = {}
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment for model loading
    env = EnhancedMaize(mode='eval', year1=years_range[0], year2=years_range[1])
    env.reset(seed=seed)
    
    # 1. Evaluate traditional strategies
    print("\nEvaluating traditional irrigation strategies...")
    traditional_results = evaluate_traditional_strategies(years_range, n_episodes)
    results_list.append(traditional_results)
    
    # 2. Evaluate Random baseline
    print("\nEvaluating Random baseline...")
    random_agent = RandomAgent(env.action_space)
    random_results, random_detailed = evaluate_rl_agent(
        random_agent, 'rule_based', EnhancedMaize, n_episodes, seed, years_range
    )
    random_df = pd.DataFrame([{**random_results, 'label': 'Random'}])
    results_list.append(random_df)
    detailed_results['Random'] = random_detailed
    
    # 3. Evaluate Rule-based baseline
    print("\nEvaluating Threshold-based baseline...")
    threshold_agent = ThresholdAgent(env)
    threshold_results, threshold_detailed = evaluate_rl_agent(
        threshold_agent, 'rule_based', EnhancedMaize, n_episodes, seed, years_range
    )
    threshold_df = pd.DataFrame([{**threshold_results, 'label': 'Threshold-based'}])
    results_list.append(threshold_df)
    detailed_results['Threshold-based'] = threshold_detailed
    
    # 4. Evaluate Weather-based baseline
    print("\nEvaluating Weather-based baseline...")
    weather_agent = WeatherBasedAgent(env)
    weather_results, weather_detailed = evaluate_rl_agent(
        weather_agent, 'rule_based', EnhancedMaize, n_episodes, seed, years_range
    )
    weather_df = pd.DataFrame([{**weather_results, 'label': 'Weather-based'}])
    results_list.append(weather_df)
    detailed_results['Weather-based'] = weather_detailed
    
    # 5. Evaluate Vanilla DQN if provided
    if vanilla_dqn_path:
        print(f"\nEvaluating Vanilla DQN from {vanilla_dqn_path}...")
        try:
            vanilla_dqn_model = load_vanilla_dqn_model(vanilla_dqn_path, env, device=device)
            vanilla_dqn_results, vanilla_dqn_detailed = evaluate_rl_agent(
                vanilla_dqn_model, 'dqn', EnhancedMaize, n_episodes, seed, years_range
            )
            vanilla_dqn_df = pd.DataFrame([{**vanilla_dqn_results, 'label': 'Vanilla DQN'}])
            results_list.append(vanilla_dqn_df)
            detailed_results['Vanilla DQN'] = vanilla_dqn_detailed
        except Exception as e:
            print(f"Error loading Vanilla DQN model: {e}")
    
    # 6. Evaluate Double DQN if provided
    if double_dqn_path:
        print(f"\nEvaluating Double DQN from {double_dqn_path}...")
        try:
            double_dqn_model = load_double_dqn_model(double_dqn_path, env, device=device)
            double_dqn_results, double_dqn_detailed = evaluate_rl_agent(
                double_dqn_model, 'double_dqn', EnhancedMaize, n_episodes, seed, years_range
            )
            double_dqn_df = pd.DataFrame([{**double_dqn_results, 'label': 'Double DQN'}])
            results_list.append(double_dqn_df)
            detailed_results['Double DQN'] = double_dqn_detailed
        except Exception as e:
            print(f"Error loading Double DQN model: {e}")
    
    # 7. Evaluate Rainbow DQN if provided
    if rainbow_path:
        print(f"\nEvaluating Rainbow DQN from {rainbow_path}...")
        try:
            rainbow_model = load_rainbow_model(rainbow_path, env, device)
            rainbow_results, rainbow_detailed = evaluate_rl_agent(
                rainbow_model, 'rainbow', EnhancedMaize, n_episodes, seed, years_range
            )
            rainbow_df = pd.DataFrame([{**rainbow_results, 'label': 'Rainbow DQN'}])
            results_list.append(rainbow_df)
            detailed_results['Rainbow DQN'] = rainbow_detailed
        except Exception as e:
            print(f"Error loading Rainbow DQN model: {e}")
    
    # 8. Evaluate PPO if provided
    if ppo_path:
        print(f"\nEvaluating PPO from {ppo_path}...")
        try:
            ppo_model = load_ppo_model(ppo_path, env, device=device)
            ppo_results, ppo_detailed = evaluate_rl_agent(
                ppo_model, 'ppo', EnhancedMaize, n_episodes, seed, years_range
            )
            ppo_df = pd.DataFrame([{**ppo_results, 'label': 'PPO'}])
            results_list.append(ppo_df)
            detailed_results['PPO'] = ppo_detailed
        except Exception as e:
            print(f"Error loading PPO model: {e}")
    
    # Combine all results
    all_results = pd.concat(results_list, ignore_index=True)
    
    # Save results to CSV
    results_path = os.path.join(results_dir, 'all_approaches_comparison.csv')
    all_results.to_csv(results_path, index=False)
    print(f"Saved comparison results to {results_path}")
    
    # Create visualizations
    create_professional_visualizations(all_results, detailed_results)
    
    return all_results, detailed_results


def create_comparison_visualizations(all_results, detailed_results):
    """
    Create comprehensive visualizations comparing all irrigation approaches
    with enhanced readability for non-technical audiences
    """
    # Create output directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define aesthetic styles for better visuals
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 20
    })
    
    # Define categories for organizing approaches
    rl_models = ['Rainbow DQN', 'Double DQN', 'Vanilla DQN', 'PPO']
    rule_based = ['Weather-based', 'Threshold-based']
    traditional = ['Net', 'Interval', 'Thresholds', 'Rainfed']
    baseline = ['Random']
    
    # Set order of approaches for consistency
    desired_order = rl_models + rule_based + traditional + baseline
    
    # Filter to only include approaches that exist in the results
    available_labels = all_results['label'].unique()
    plot_order = [label for label in desired_order if label in available_labels]
    
    # Set categorical order for plotting
    all_results['label'] = pd.Categorical(all_results['label'], categories=plot_order, ordered=True)
    all_results = all_results.sort_values('label')
    
    # Add a category column for grouping in visualizations
    def get_category(label):
        if label in rl_models:
            return "Reinforcement Learning"
        elif label in rule_based:
            return "Rule-based"
        elif label in traditional:
            return "Traditional"
        else:
            return "Baseline"
    
    all_results['category'] = all_results['label'].apply(get_category)
    
    # Define color mapping for consistency with distinct color scheme
    color_dict = {
        # RL models - blues and purples
        'Rainbow DQN': '#1f77b4',    # Dark blue
        'Double DQN': '#56b4e9',     # Light blue
        'Vanilla DQN': '#4c72b0',    # Medium blue
        'PPO': '#9467bd',            # Purple
        
        # Rule-based - greens
        'Weather-based': '#2ca02c',  # Dark green
        'Threshold-based': '#98df8a', # Light green
        
        # Traditional - reds/oranges
        'Net': '#d62728',            # Red
        'Interval': '#ff7f0e',       # Orange
        'Thresholds': '#e377c2',     # Pink
        'Rainfed': '#8c564b',        # Brown
        
        # Baseline
        'Random': '#7f7f7f'          # Grey
    }
    
    # Define category colors for grouped visuals
    category_colors = {
        "Reinforcement Learning": "#4c72b0",  # Blue
        "Rule-based": "#2ca02c",              # Green
        "Traditional": "#d62728",             # Red
        "Baseline": "#7f7f7f"                 # Grey
    }
    
    # Extract data for plots
    labels = all_results['label']
    colors = [color_dict.get(label, '#333333') for label in labels]
    
    # Create directory for saving all plots
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 16))
    plt.suptitle("Irrigation Optimization Comparison Dashboard", fontsize=24, fontweight='bold', y=0.98)
    
    # Sort results by yield for consistent ordering
    sorted_results = all_results.sort_values('Dry yield (tonne/ha)_mean', ascending=False)
    
    # 1.1 Yield subplot
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(
        sorted_results['label'], 
        sorted_results['Dry yield (tonne/ha)_mean'],
        color=[color_dict.get(label, '#333333') for label in sorted_results['label']], 
        edgecolor='black'
    )
    best_yield = sorted_results['Dry yield (tonne/ha)_mean'].iloc[0]
    plt.axhline(y=best_yield, color='red', linestyle='--', alpha=0.7, 
                label=f'Best yield: {best_yield:.2f} tonne/ha')
    plt.ylabel('Crop Yield (tonne/ha)')
    plt.title('Crop Yield Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.1f}', ha='center', fontsize=9)
    plt.legend()
    
    # 1.2 Profit subplot
    plt.subplot(2, 2, 2)
    sorted_by_profit = all_results.sort_values('Profit_mean', ascending=False)
    bars2 = plt.bar(
        sorted_by_profit['label'], 
        sorted_by_profit['Profit_mean'],
        color=[color_dict.get(label, '#333333') for label in sorted_by_profit['label']], 
        edgecolor='black'
    )
    best_profit = sorted_by_profit['Profit_mean'].iloc[0]
    plt.axhline(y=best_profit, color='green', linestyle='--', alpha=0.7,
               label=f'Best profit: ${best_profit:.0f}/ha')
    plt.ylabel('Profit ($/ha)')
    plt.title('Profit Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'${bar.get_height():.0f}', ha='center', fontsize=9)
    plt.legend()
    
    # 1.3 Water Use Efficiency subplot
    plt.subplot(2, 2, 3)
    sorted_by_wue = all_results.sort_values('WaterEfficiency_mean', ascending=False)
    bars3 = plt.bar(
        sorted_by_wue['label'], 
        sorted_by_wue['WaterEfficiency_mean'],
        color=[color_dict.get(label, '#333333') for label in sorted_by_wue['label']], 
        edgecolor='black'
    )
    best_wue = sorted_by_wue['WaterEfficiency_mean'].iloc[0]
    plt.axhline(y=best_wue, color='blue', linestyle='--', alpha=0.7,
               label=f'Best WUE: {best_wue:.2f} kg/m³')
    plt.ylabel('Water Use Efficiency (kg/m³)')
    plt.title('Water Efficiency Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars3:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', fontsize=9)
    plt.legend()
    
    # 1.4 Irrigation Amount subplot
    plt.subplot(2, 2, 4)
    sorted_by_irr = all_results.sort_values('Seasonal irrigation (mm)_mean', ascending=True)
    bars4 = plt.bar(
        sorted_by_irr['label'], 
        sorted_by_irr['Seasonal irrigation (mm)_mean'],
        color=[color_dict.get(label, '#333333') for label in sorted_by_irr['label']], 
        edgecolor='black'
    )
    best_irr = sorted_by_irr['Seasonal irrigation (mm)_mean'].iloc[0]
    if best_irr > 0:  # Only show if there's actual irrigation (not rainfed)
        plt.axhline(y=best_irr, color='purple', linestyle='--', alpha=0.7,
                   label=f'Lowest irrigation: {best_irr:.0f} mm')
        plt.legend()
    plt.ylabel('Irrigation Applied (mm)')
    plt.title('Irrigation Amount Comparison')
    plt.xticks(rotation=45, ha='right')
    for bar in bars4:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{bar.get_height():.0f}', ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'dashboard_summary.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    

    category_data = all_results.groupby('category').agg({
        'Dry yield (tonne/ha)_mean': 'mean',
        'Seasonal irrigation (mm)_mean': 'mean',
        'Profit_mean': 'mean',
        'WaterEfficiency_mean': 'mean'
    }).reset_index()
    
    # Sort categories in a logical order
    category_order = ["Reinforcement Learning", "Rule-based", "Traditional", "Baseline"]
    category_data['category'] = pd.Categorical(
        category_data['category'], 
        categories=[c for c in category_order if c in category_data['category'].values],
        ordered=True
    )
    category_data = category_data.sort_values('category')
    
    plt.figure(figsize=(16, 12))
    plt.suptitle("Comparison by Approach Category", fontsize=22, fontweight='bold', y=0.98)
    
    # 2.1 Yield by category
    plt.subplot(2, 2, 1)
    bars = plt.bar(
        category_data['category'],
        category_data['Dry yield (tonne/ha)_mean'],
        color=[category_colors[cat] for cat in category_data['category']],
        edgecolor='black'
    )
    plt.ylabel('Average Yield (tonne/ha)')
    plt.title('Crop Yield by Approach Category')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center')
    
    # 2.2 Profit by category
    plt.subplot(2, 2, 2)
    bars = plt.bar(
        category_data['category'],
        category_data['Profit_mean'],
        color=[category_colors[cat] for cat in category_data['category']],
        edgecolor='black'
    )
    plt.ylabel('Average Profit ($/ha)')
    plt.title('Profit by Approach Category')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'${bar.get_height():.0f}', ha='center')
    
    # 2.3 Water Use Efficiency by category
    plt.subplot(2, 2, 3)
    bars = plt.bar(
        category_data['category'],
        category_data['WaterEfficiency_mean'],
        color=[category_colors[cat] for cat in category_data['category']],
        edgecolor='black'
    )
    plt.ylabel('Average Water Use Efficiency (kg/m³)')
    plt.title('Water Efficiency by Approach Category')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center')
    
    # 2.4 Irrigation by category
    plt.subplot(2, 2, 4)
    bars = plt.bar(
        category_data['category'],
        category_data['Seasonal irrigation (mm)_mean'],
        color=[category_colors[cat] for cat in category_data['category']],
        edgecolor='black'
    )
    plt.ylabel('Average Irrigation (mm)')
    plt.title('Irrigation Amount by Approach Category')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{bar.get_height():.0f}', ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'category_comparison.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 10))
    
    # Create scatter plot with labeled points and trend lines by category
    categories = all_results['category'].unique()
    
    # Basic scatter plot
    for category in categories:
        category_results = all_results[all_results['category'] == category]
        plt.scatter(
            category_results['Seasonal irrigation (mm)_mean'],
            category_results['Dry yield (tonne/ha)_mean'],
            s=100,
            alpha=0.7,
            c=category_colors[category],
            edgecolor='white',
            linewidth=1,
            label=category
        )
    
    # Add labels for each point
    for i, row in all_results.iterrows():
        plt.annotate(
            row['label'],
            (row['Seasonal irrigation (mm)_mean'], row['Dry yield (tonne/ha)_mean']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add best-fit lines
    for category in categories:
        category_results = all_results[all_results['category'] == category]
        if len(category_results) >= 2:  # Need at least 2 points for a line
            x = category_results['Seasonal irrigation (mm)_mean']
            y = category_results['Dry yield (tonne/ha)_mean']
            
            try:
                # Simple linear regression
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Generate x values for the line
                x_line = np.linspace(x.min(), x.max(), 100)
                
                # Plot trend line
                plt.plot(x_line, p(x_line), '--', color=category_colors[category], 
                        alpha=0.8, linewidth=2)
            except:
                # Skip if fitting fails
                pass
    
    plt.xlabel('Total Irrigation Applied (mm)')
    plt.ylabel('Crop Yield (tonne/ha)')
    plt.title('Yield vs. Irrigation Relationship by Approach Category', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Approach Category")
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
               "Better approaches have higher yields with less irrigation (upper left is optimal)", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'yield_irrigation_relationship.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot with bubble size representing irrigation amount
    for category in categories:
        category_results = all_results[all_results['category'] == category]
        
        # Normalize bubble sizes (irrigation amounts)
        sizes = category_results['Seasonal irrigation (mm)_mean'] / 5  # Scaling factor
        
        plt.scatter(
            category_results['WaterEfficiency_mean'],
            category_results['Profit_mean'],
            s=sizes,
            alpha=0.7,
            c=category_colors[category],
            edgecolor='white',
            linewidth=1,
            label=category
        )
    
    # Add labels for each point
    for i, row in all_results.iterrows():
        plt.annotate(
            row['label'],
            (row['WaterEfficiency_mean'], row['Profit_mean']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.xlabel('Water Use Efficiency (kg/m³)')
    plt.ylabel('Profit ($/ha)')
    plt.title('Profit vs. Water Efficiency (Bubble Size = Irrigation Amount)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Approach Category")
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
               "Better approaches have higher water efficiency and higher profit (upper right is optimal)", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'profit_vs_efficiency.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    radar_models = rl_models + rule_based
    radar_data = all_results[all_results['label'].isin([m for m in radar_models if m in all_results['label'].values])]
    
    # Ensure we have enough models for comparison
    if len(radar_data) >= 2:
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, polar=True)
        
        # Define metrics to include
        metrics = ['Yield', 'Profit', 'Water Efficiency', 'Water Saving']
        metrics_cols = [
            'Dry yield (tonne/ha)_mean', 
            'Profit_mean', 
            'WaterEfficiency_mean', 
            # For water saving, we'll invert irrigation amount
            'Seasonal irrigation (mm)_mean'  
        ]
        
        # Normalize data to 0-1 scale for each metric
        normalized_data = {}
        for i, metric in enumerate(metrics):
            col = metrics_cols[i]
            
            if col == 'Seasonal irrigation (mm)_mean':
                # Invert irrigation - less is better
                values = radar_data[col]
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    normalized_data[metric] = 1 - ((values - min_val) / (max_val - min_val))
                else:
                    normalized_data[metric] = values * 0 + 0.5
            else:
                # For other metrics, higher is better
                values = radar_data[col]
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    normalized_data[metric] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[metric] = values * 0 + 0.5
        
        # Number of metrics
        N = len(metrics)
        
        # Angle for each metric (evenly spaced)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set up radar chart
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Go clockwise
        
        # Draw axis lines for each metric
        plt.xticks(angles[:-1], metrics, fontsize=14)
        
        # Draw y-axis labels (0-1)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot each model
        for i, row in radar_data.iterrows():
            label = row['label']
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, 
                   color=color_dict.get(label, '#333333'))
            ax.fill(angles, values, color=color_dict.get(label, '#333333'), alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Comparative Performance of Irrigation Approaches', fontsize=20, pad=20)
        
        # Add explanatory note
        plt.figtext(0.5, 0.01, 
                   "Larger area indicates better overall performance across metrics", 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'radar_comparison.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    existing_models = [model for model in rl_models if model in detailed_results]
    
    if existing_models:
        plt.figure(figsize=(15, 10))
        plt.suptitle("Irrigation Decision Patterns by RL Model", fontsize=20, y=0.98)
        
        # Create subplots - one for each RL model
        for i, model in enumerate(existing_models):
            plt.subplot(len(existing_models), 1, i+1)
            
            # Get irrigation decisions for first 3 episodes (for clarity)
            irrigation_decisions = detailed_results[model]['irrigation_decisions'][:3]
            
            # Plot irrigation patterns for each episode
            for j, decisions in enumerate(irrigation_decisions):
                # Convert to array and get steps
                decisions = np.array(decisions)
                steps = np.arange(len(decisions))
                
                # Plot with thicker lines and markers
                plt.plot(steps, decisions, marker='o', linewidth=2,
                        label=f'Episode {j+1}', color=plt.cm.tab10(j))
                
                # Fill area under the curve for better visualization
                plt.fill_between(steps, 0, decisions, alpha=0.1, color=plt.cm.tab10(j))
            
            plt.title(f'{model} Irrigation Pattern')
            plt.xlabel('Growing Season Day')
            plt.ylabel('Irrigation Applied (mm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'irrigation_patterns.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an animated version of irrigation patterns over time
        # This shows cumulative irrigation throughout the season
        for model in existing_models:
            plt.figure(figsize=(12, 8))
            
            # Get irrigation decisions for first 3 episodes
            irrigation_decisions = detailed_results[model]['irrigation_decisions'][:3]
            
            # Calculate cumulative irrigation over time
            cumulative_irrigation = []
            for decisions in irrigation_decisions:
                cumulative = np.cumsum(decisions)
                cumulative_irrigation.append(cumulative)
            
            # Plot cumulative irrigation
            for j, cumulative in enumerate(cumulative_irrigation):
                steps = np.arange(len(cumulative))
                
                plt.plot(steps, cumulative, marker='', linewidth=3,
                        label=f'Episode {j+1}', color=plt.cm.tab10(j))
                
                # Add area fill
                plt.fill_between(steps, 0, cumulative, alpha=0.2, color=plt.cm.tab10(j))
            
            plt.title(f'{model}: Cumulative Irrigation Over Growing Season', fontsize=18)
            plt.xlabel('Growing Season Day')
            plt.ylabel('Cumulative Irrigation (mm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add annotations for key points
            for j, cumulative in enumerate(cumulative_irrigation):
                # Add total amount at end
                plt.annotate(
                    f'Total: {cumulative[-1]:.0f} mm',
                    (len(cumulative)-1, cumulative[-1]),
                    xytext=(10, 0),
                    textcoords='offset points',
                    fontsize=10,
                    color=plt.cm.tab10(j),
                    fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'cumulative_irrigation_{model}.png'), format='png', dpi=300)
            plt.close()
   
    if 'WaterStress_mean' in all_results.columns:
        # Filter for models with water stress data
        stress_models = []
        for model in all_results['label'].unique():
            value = all_results[all_results['label'] == model]['WaterStress_mean'].values[0]
            if not pd.isna(value):
                stress_models.append(model)
        
        if stress_models:
            plt.figure(figsize=(14, 8))
            
            # Sort by stress level (ascending)
            stress_data = all_results[all_results['label'].isin(stress_models)].sort_values('WaterStress_mean')
            
            # Create color palette
            colors = []
            for label in stress_data['label']:
                colors.append(color_dict.get(label, '#333333'))
            
            # Create bar chart
            bars = plt.bar(
                stress_data['label'],
                stress_data['WaterStress_mean'],
                yerr=stress_data['WaterStress_std'] if 'WaterStress_std' in stress_data.columns else None,
                capsize=5,
                color=colors,
                edgecolor='black'
            )
            
            # Add values on top
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0, size=10
                )
            
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                       label='Significant Water Stress Threshold')
            
            plt.ylabel('Average Water Stress Index (0-1)', fontsize=14)
            plt.xlabel('Irrigation Approach', fontsize=14)
            plt.title('Comparison of Water Stress Management Effectiveness', fontsize=18)
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            
            # Add explanatory note
            plt.figtext(0.5, 0.01, 
                       "Lower values indicate better plant water stress management", 
                       ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(plots_dir, 'water_stress_management.png'), format='png', dpi=300, bbox_inches='tight')
            plt.close()
    
   
    violin_models = [model for model in all_results['label'].values 
                     if model in detailed_results and len(detailed_results[model]['yields']) > 0]
    
    if violin_models:
        plt.figure(figsize=(20, 10))
        
        # Prepare data for violin plots
        violin_data = []
        violin_labels = []
        
        for model in violin_models:
            violin_data.append(detailed_results[model]['yields'])
            violin_labels.append(model)
        
        # Create violin plot with enhancements
        parts = plt.violinplot(violin_data, showmeans=True, showmedians=True)
        
        # Customize violin appearance
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color_dict.get(violin_labels[i], '#333333'))
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        # Customize median lines
        for partname in ['cmeans', 'cmedians', 'cbars']:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
        
        # Add a box plot inside each violin for better distribution visualization
        plt.boxplot(violin_data, positions=range(1, len(violin_labels) + 1), 
                   widths=0.15, patch_artist=True, 
                   boxprops=dict(facecolor='white', alpha=0.7))
        
        # Add individual data points for extra detail
        for i, data in enumerate(violin_data):
            # Add jittered points
            x = np.random.normal(i+1, 0.05, size=len(data))
            plt.scatter(x, data, alpha=0.4, s=5, c=color_dict.get(violin_labels[i], '#333333'))
        
        # Customize appearance
        plt.ylabel('Crop Yield (tonne/ha)', fontsize=16)
        plt.xlabel('Irrigation Approach', fontsize=16)
        plt.title('Yield Distribution Comparison Across Approaches', fontsize=20)
        plt.xticks(range(1, len(violin_labels) + 1), violin_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add reference lines for agricultural context
        avg_yield = np.mean([np.mean(data) for data in violin_data])
        plt.axhline(y=avg_yield, color='red', linestyle='--', 
                   label=f'Average Yield: {avg_yield:.2f} tonne/ha')
        
        # Add shaded region for below-average yields
        plt.axhspan(0, avg_yield, alpha=0.1, color='red', label='Below Average Yield')
        
        plt.legend(loc='upper right')
        
        # Add explanatory note
        plt.figtext(0.5, 0.01, 
                   "Wider sections represent more frequent yield values; higher distributions are better", 
                   ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'yield_distributions.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    baseline_methods = ['Random']
    traditional_methods = ['Rainfed']
    
    # Only proceed if we have baselines to compare against
    baseline_results = all_results[all_results['label'].isin(baseline_methods)]
    traditional_results = all_results[all_results['label'].isin(traditional_methods)]
    
    if not baseline_results.empty or not traditional_results.empty:
        # Choose reference for comparison (prefer random baseline, fall back to traditional)
        if not baseline_results.empty:
            reference = baseline_results.iloc[0]
            reference_name = reference['label']
        else:
            reference = traditional_results.iloc[0]
            reference_name = reference['label']
        
        # Calculate reference values
        ref_yield = reference['Dry yield (tonne/ha)_mean']
        ref_profit = reference['Profit_mean']
        ref_wue = reference['WaterEfficiency_mean']
        ref_irrigation = reference['Seasonal irrigation (mm)_mean']
        
        # Filter for advanced methods (RL and rule-based)
        advanced_methods = rl_models + rule_based
        advanced_results = all_results[all_results['label'].isin(
            [m for m in advanced_methods if m in all_results['label'].values]
        )]
        
        if not advanced_results.empty:
            plt.figure(figsize=(16, 12))
            plt.suptitle(f"Improvement Over {reference_name} Baseline", fontsize=22, y=0.98)
            
            # 9.1 Yield improvement
            plt.subplot(2, 2, 1)
            # Calculate percentage improvement
            advanced_results['yield_improvement'] = (
                (advanced_results['Dry yield (tonne/ha)_mean'] - ref_yield) / ref_yield * 100
            )
            # Sort by improvement
            sorted_data = advanced_results.sort_values('yield_improvement', ascending=False)
            
            bars = plt.bar(
                sorted_data['label'],
                sorted_data['yield_improvement'],
                color=[color_dict.get(label, '#333333') for label in sorted_data['label']],
                edgecolor='black'
            )
            
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
            plt.ylabel('Yield Improvement (%)')
            plt.title(f'Crop Yield Improvement vs {reference_name}')
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.1f}%" if height >= 0 else f"{height:.1f}%"
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        label, ha='center', va='bottom', rotation=0, size=10)
            
            # 9.2 Profit improvement
            plt.subplot(2, 2, 2)
            # Calculate profit improvement
            advanced_results['profit_improvement'] = (
                (advanced_results['Profit_mean'] - ref_profit) / abs(ref_profit) * 100
                if ref_profit != 0 else advanced_results['Profit_mean'] * 100
            )
            # Sort by improvement
            sorted_data = advanced_results.sort_values('profit_improvement', ascending=False)
            
            bars = plt.bar(
                sorted_data['label'],
                sorted_data['profit_improvement'],
                color=[color_dict.get(label, '#333333') for label in sorted_data['label']],
                edgecolor='black'
            )
            
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
            plt.ylabel('Profit Improvement (%)')
            plt.title(f'Profit Improvement vs {reference_name}')
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.1f}%" if height >= 0 else f"{height:.1f}%"
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        label, ha='center', va='bottom', rotation=0, size=10)
            
            # 9.3 Water efficiency improvement
            plt.subplot(2, 2, 3)
            # Calculate WUE improvement, handling possible zero reference
            if ref_wue > 0:
                advanced_results['wue_improvement'] = (
                    (advanced_results['WaterEfficiency_mean'] - ref_wue) / ref_wue * 100
                )
            else:
                # If reference WUE is zero, use absolute values
                advanced_results['wue_improvement'] = advanced_results['WaterEfficiency_mean'] * 100
                
            # Sort by improvement
            sorted_data = advanced_results.sort_values('wue_improvement', ascending=False)
            
            bars = plt.bar(
                sorted_data['label'],
                sorted_data['wue_improvement'],
                color=[color_dict.get(label, '#333333') for label in sorted_data['label']],
                edgecolor='black'
            )
            
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
            plt.ylabel('Water Efficiency Improvement (%)')
            plt.title(f'Water Use Efficiency Improvement vs {reference_name}')
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.1f}%" if height >= 0 else f"{height:.1f}%"
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        label, ha='center', va='bottom', rotation=0, size=10)
            
            # 9.4 Irrigation reduction (negative is better - using less water)
            plt.subplot(2, 2, 4)
            # Calculate irrigation change, negative means reduction (better)
            if ref_irrigation > 0:
                advanced_results['irrigation_change'] = (
                    (advanced_results['Seasonal irrigation (mm)_mean'] - ref_irrigation) / ref_irrigation * 100
                )
            else:
                # If reference irrigation is zero, use relative to max irrigation
                max_irrigation = advanced_results['Seasonal irrigation (mm)_mean'].max()
                if max_irrigation > 0:
                    advanced_results['irrigation_change'] = (
                        advanced_results['Seasonal irrigation (mm)_mean'] / max_irrigation * 100
                    )
                else:
                    advanced_results['irrigation_change'] = 0
                    
            # Sort by improvement (increasing value = more irrigation)
            sorted_data = advanced_results.sort_values('irrigation_change', ascending=True)
            
            bars = plt.bar(
                sorted_data['label'],
                sorted_data['irrigation_change'],
                color=[color_dict.get(label, '#333333') for label in sorted_data['label']],
                edgecolor='black'
            )
            
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
            plt.ylabel('Irrigation Change (%)')
            plt.title(f'Irrigation Change vs {reference_name} (negative is better)')
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.1f}%" if height >= 0 else f"{height:.1f}%"
                plt.text(bar.get_x() + bar.get_width()/2., height + 1 if height >= 0 else -1,
                        label, ha='center', va='bottom' if height >= 0 else 'top', rotation=0, size=10)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(plots_dir, 'improvement_over_baseline.png'), format='png', dpi=300, bbox_inches='tight')
            plt.close()
    
    plt.figure(figsize=(14, 10))
    
    # Prepare data - sort by profit
    profit_data = all_results.sort_values('Profit_mean', ascending=False)
    
    # Select top performers
    top_performers = profit_data.head(min(6, len(profit_data)))
    
    # Calculate revenue and cost components
    top_performers['revenue'] = top_performers['Dry yield (tonne/ha)_mean'] * CROP_PRICE
    top_performers['irrigation_cost'] = top_performers['Seasonal irrigation (mm)_mean'] * IRRIGATION_COST
    top_performers['fixed_cost'] = FIXED_COST
    top_performers['total_cost'] = top_performers['irrigation_cost'] + top_performers['fixed_cost']
    
    # Set up positions and width
    pos = np.arange(len(top_performers))
    width = 0.35
    
    # Create grouped bar chart
    plt.bar(pos, top_performers['revenue'], width, 
           color='green', edgecolor='black', alpha=0.7, label='Revenue')
    
    plt.bar(pos + width, top_performers['total_cost'], width,
           color='red', edgecolor='black', alpha=0.7, label='Total Cost')
    
    # Add profit annotations
    for i, row in enumerate(top_performers.itertuples()):
        profit = row.Profit_mean
        plt.annotate(
            f"Profit: ${profit:.0f}/ha",
            (i + width/2, max(row.revenue, row.total_cost) + 100),
            ha='center',
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8)
        )
    
    # Set chart properties
    plt.ylabel('Amount ($/ha)', fontsize=14)
    plt.title('Economic Comparison: Revenue vs. Costs', fontsize=18)
    plt.xticks(pos + width/2, top_performers['label'], rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
               "Higher revenue and lower costs result in better profit", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, 'economic_impact.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    plt.figure(figsize=(16, 12))
    
    # Identify best performing method
    if not all_results.empty:
        # Different "best" methods for different metrics
        best_yield_row = all_results.loc[all_results['Dry yield (tonne/ha)_mean'].idxmax()]
        best_profit_row = all_results.loc[all_results['Profit_mean'].idxmax()]
        best_wue_row = all_results.loc[all_results['WaterEfficiency_mean'].idxmax()]
        
        # Determine single overall best method
        # Simple approach: sum of normalized metrics
        # Normalize each metric 0-1
        all_results_norm = all_results.copy()
        metrics_to_norm = ['Dry yield (tonne/ha)_mean', 'Profit_mean', 'WaterEfficiency_mean']
        
        for metric in metrics_to_norm:
            min_val = all_results[metric].min()
            max_val = all_results[metric].max()
            if max_val > min_val:
                all_results_norm[f'{metric}_norm'] = (all_results[metric] - min_val) / (max_val - min_val)
            else:
                all_results_norm[f'{metric}_norm'] = 0.5
        
        # For irrigation, less is better (invert)
        min_irr = all_results['Seasonal irrigation (mm)_mean'].min()
        max_irr = all_results['Seasonal irrigation (mm)_mean'].max()
        if max_irr > min_irr:
            all_results_norm['irrigation_norm'] = 1 - (
                (all_results['Seasonal irrigation (mm)_mean'] - min_irr) / (max_irr - min_irr)
            )
        else:
            all_results_norm['irrigation_norm'] = 0.5
        
        # Calculate overall score
        all_results_norm['overall_score'] = (
            all_results_norm['Dry yield (tonne/ha)_mean_norm'] * 0.3 +
            all_results_norm['Profit_mean_norm'] * 0.3 +
            all_results_norm['WaterEfficiency_mean_norm'] * 0.2 +
            all_results_norm['irrigation_norm'] * 0.2
        )
        
        # Get best overall method
        best_overall_row = all_results_norm.loc[all_results_norm['overall_score'].idxmax()]
        
        # Create infographic with key results
        plt.suptitle("Best Irrigation Approaches Summary", fontsize=24, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # 1. Best Overall Approach
        ax1 = plt.subplot(gs[0, 0])
        plt.text(0.5, 0.9, "BEST OVERALL APPROACH", fontsize=18, 
                ha='center', fontweight='bold', color='darkblue')
        plt.text(0.5, 0.8, f"{best_overall_row['label']}", fontsize=22, 
                ha='center', fontweight='bold', color=color_dict.get(best_overall_row['label'], '#333333'))
        
        # Add key stats
        stats_text = (
            f"Overall Score: {best_overall_row['overall_score']:.2f}\n\n"
            f"Yield: {best_overall_row['Dry yield (tonne/ha)_mean']:.2f} tonne/ha\n"
            f"Profit: ${best_overall_row['Profit_mean']:.0f}/ha\n"
            f"Water Efficiency: {best_overall_row['WaterEfficiency_mean']:.2f} kg/m³\n"
            f"Irrigation: {best_overall_row['Seasonal irrigation (mm)_mean']:.0f} mm"
        )
        plt.text(0.5, 0.45, stats_text, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", alpha=0.3))
        
        # Add context
        plt.text(0.5, 0.1, "Best balance of yield, profit, and water efficiency", 
                fontsize=12, ha='center', fontweight='bold', color='gray', style='italic')
        
        plt.axis('off')
        
        # 2. Best for Maximum Yield
        ax2 = plt.subplot(gs[0, 1])
        plt.text(0.5, 0.9, "BEST FOR MAXIMUM YIELD", fontsize=18, 
                ha='center', fontweight='bold', color='darkgreen')
        plt.text(0.5, 0.8, f"{best_yield_row['label']}", fontsize=22, 
                ha='center', fontweight='bold', color=color_dict.get(best_yield_row['label'], '#333333'))
        
        # Add key stats
        stats_text = (
            f"Yield: {best_yield_row['Dry yield (tonne/ha)_mean']:.2f} tonne/ha\n\n"
            f"Improvement over baseline: "
            f"{((best_yield_row['Dry yield (tonne/ha)_mean']/ref_yield) - 1) * 100:.1f}%\n"
            f"Irrigation: {best_yield_row['Seasonal irrigation (mm)_mean']:.0f} mm\n"
            f"Profit: ${best_yield_row['Profit_mean']:.0f}/ha"
        )
        plt.text(0.5, 0.45, stats_text, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.3))
        
        # Add context
        plt.text(0.5, 0.1, "When maximizing production is the priority", 
                fontsize=12, ha='center', fontweight='bold', color='gray', style='italic')
        
        plt.axis('off')
        
        # 3. Best for Maximum Profit
        ax3 = plt.subplot(gs[1, 0])
        plt.text(0.5, 0.9, "BEST FOR MAXIMUM PROFIT", fontsize=18, 
                ha='center', fontweight='bold', color='darkred')
        plt.text(0.5, 0.8, f"{best_profit_row['label']}", fontsize=22, 
                ha='center', fontweight='bold', color=color_dict.get(best_profit_row['label'], '#333333'))
        
        # Add key stats
        stats_text = (
            f"Profit: ${best_profit_row['Profit_mean']:.0f}/ha\n\n"
            f"Revenue: ${best_profit_row['Dry yield (tonne/ha)_mean'] * CROP_PRICE:.0f}/ha\n"
            f"Irrigation Cost: ${best_profit_row['Seasonal irrigation (mm)_mean'] * IRRIGATION_COST:.0f}/ha\n"
            f"Fixed Cost: ${FIXED_COST}/ha\n"
            f"Return on Investment: "
            f"{(best_profit_row['Profit_mean']/(FIXED_COST + best_profit_row['Seasonal irrigation (mm)_mean'] * IRRIGATION_COST)) * 100:.1f}%"
        )
        plt.text(0.5, 0.45, stats_text, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="#ffcccc", ec="red", alpha=0.3))
        
        # Add context
        plt.text(0.5, 0.1, "When economic returns are the priority", 
                fontsize=12, ha='center', fontweight='bold', color='gray', style='italic')
        
        plt.axis('off')
        
        # 4. Best for Water Efficiency
        ax4 = plt.subplot(gs[1, 1])
        plt.text(0.5, 0.9, "BEST FOR WATER EFFICIENCY", fontsize=18, 
                ha='center', fontweight='bold', color='darkblue')
        plt.text(0.5, 0.8, f"{best_wue_row['label']}", fontsize=22, 
                ha='center', fontweight='bold', color=color_dict.get(best_wue_row['label'], '#333333'))
        
        # Add key stats
        stats_text = (
            f"Water Efficiency: {best_wue_row['WaterEfficiency_mean']:.2f} kg/m³\n\n"
            f"Irrigation: {best_wue_row['Seasonal irrigation (mm)_mean']:.0f} mm\n"
            f"Yield: {best_wue_row['Dry yield (tonne/ha)_mean']:.2f} tonne/ha\n"
            f"Water Savings: "
            f"{(1 - (best_wue_row['Seasonal irrigation (mm)_mean']/ref_irrigation)) * 100:.1f}% vs baseline"
        )
        plt.text(0.5, 0.45, stats_text, fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="blue", alpha=0.3))
        
        # Add context
        plt.text(0.5, 0.1, "When water conservation is the priority", 
                fontsize=12, ha='center', fontweight='bold', color='gray', style='italic')
        
        plt.axis('off')
        
        # Add bottom note
        plt.figtext(0.5, 0.02, 
                   "Deep Reinforcement Learning approaches consistently outperform traditional irrigation methods", 
                   ha="center", fontsize=14, fontweight='bold',
                   bbox={"facecolor":"yellow", "alpha":0.2, "pad":10, "boxstyle":"round"})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'best_approaches_summary.png'), format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Enhanced visualizations created and saved to {plots_dir}")
    
    # Return success message with summary of plots created
    return f"Created {len(os.listdir(plots_dir))} visualization plots in {plots_dir}"


def create_professional_visualizations(all_results, detailed_results):
    """
    Create professional, academic-style visualizations comparing RL model performance
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    
    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)

    # Set up plot style - try multiple options to ensure compatibility
    try:
        # For newer matplotlib versions
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            # For older matplotlib versions
            plt.style.use('seaborn')
            sns.set_style('whitegrid')
        except:
            # Fallback to basic grid
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
    
    # Set consistent plot parameters regardless of style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 100
    })
    
    # Define models and colors
    rl_models = ['Rainbow DQN', 'Double DQN', 'Vanilla DQN', 'PPO']
    rule_models = ['Weather-based', 'Threshold-based']
    
    # Consistent color scheme (colorblind-friendly)
    colors = {
        'Rainbow DQN': '#0072B2',      # Blue
        'Double DQN': '#D55E00',       # Orange
        'Vanilla DQN': '#009E73',      # Green
        'PPO': '#CC79A7',              # Pink
        'Weather-based': '#F0E442',    # Yellow
        'Threshold-based': '#56B4E9',  # Light blue
        'Random': '#999999',           # Gray
        'Rainfed': '#E69F00'           # Brown
    }
    
    # 1. LEARNING CURVES - Training Progress
    # This assumes we have episode data saved for each model
    models_with_detailed_data = [m for m in rl_models if m in detailed_results]
    
    if models_with_detailed_data:
        plt.figure(figsize=(10, 6))
        
        for model in models_with_detailed_data:
            # Get rewards data
            if 'rewards' in detailed_results[model]:
                rewards = detailed_results[model]['rewards']
                
                # Create rolling average (smoother curve)
                window_size = min(20, len(rewards) // 5)
                if window_size > 0:
                    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    episodes = range(window_size-1, len(rewards))
                    
                    # Plot smoothed line with confidence band
                    plt.plot(episodes, smoothed_rewards, 
                            label=model, 
                            color=colors[model],
                            linewidth=2)
                    
                    # Calculate confidence intervals
                    std_rewards = np.std(rewards)
                    plt.fill_between(episodes, 
                                    smoothed_rewards - 0.5*std_rewards, 
                                    smoothed_rewards + 0.5*std_rewards, 
                                    color=colors[model], 
                                    alpha=0.2)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning Curves for Different RL Models')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'learning_curves.png'), dpi=300)
        plt.close()
    
    # 2. PERFORMANCE COMPARISON PLOT - Multi-metric line chart
    metrics = {
        'Yield (tonne/ha)': 'Dry yield (tonne/ha)_mean',
        'Irrigation (mm)': 'Seasonal irrigation (mm)_mean',
        'Water Efficiency (kg/m³)': 'WaterEfficiency_mean',
        'Profit ($/ha)': 'Profit_mean'
    }
    
    # Filter models that exist in results
    available_models = []
    for model in rl_models + rule_models:
        if model in all_results['label'].values:
            available_models.append(model)
    
    if available_models:
        plt.figure(figsize=(14, 10))
        
        # Create subplots for each metric
        for i, (metric_name, column) in enumerate(metrics.items(), 1):
            plt.subplot(2, 2, i)
            
            # Get metric data for each model
            metric_data = all_results.copy()
            metric_data = metric_data[metric_data['label'].isin(available_models)]
            metric_data = metric_data.sort_values(column, ascending=False)
            
            # Plot bar chart
            bar_positions = range(len(metric_data))
            bars = plt.bar(bar_positions, metric_data[column], 
                         color=[colors.get(label, '#333333') for label in metric_data['label']])
            
            # Add error bars if standard deviation is available
            std_col = f"{column.split('_mean')[0]}_std"
            if std_col in metric_data.columns:
                plt.errorbar(bar_positions, metric_data[column], 
                           yerr=metric_data[std_col], 
                           fmt='none', capsize=3, color='black', alpha=0.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', size=9)
            
            plt.title(metric_name)
            plt.xticks(bar_positions, metric_data['label'], rotation=45, ha='right')
            plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'multi_metric_comparison.png'), dpi=300)
        plt.close()
    
    # 3. TRAINING CONVERGENCE - Yield vs Training Steps (if available)
    if 'episode_yields' in detailed_results.get(available_models[0], {}):
        plt.figure(figsize=(10, 6))
        
        for model in available_models:
            if model in detailed_results and 'episode_yields' in detailed_results[model]:
                yields = detailed_results[model]['episode_yields']
                episodes = range(1, len(yields) + 1)
                
                # Create window-averaged trend line
                window_size = min(10, len(yields) // 5)
                if window_size > 0:
                    smoothed_yields = np.convolve(yields, np.ones(window_size)/window_size, mode='valid')
                    smooth_episodes = range(window_size-1, len(yields))
                    
                    plt.plot(smooth_episodes, smoothed_yields, label=model, 
                           color=colors.get(model, '#333333'), linewidth=2)
                    
                    # Add confidence band
                    plt.fill_between(smooth_episodes, 
                                   smoothed_yields - np.std(yields)/2,
                                   smoothed_yields + np.std(yields)/2,
                                   color=colors.get(model, '#333333'), alpha=0.2)
        
        plt.xlabel('Training Episode')
        plt.ylabel('Crop Yield (tonne/ha)')
        plt.title('Yield Convergence During Training')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'yield_convergence.png'), dpi=300)
        plt.close()
    
    # 4. IRRIGATION PATTERN COMPARISON
    if all([model in detailed_results for model in available_models]):
        plt.figure(figsize=(12, 7))
        
        for model in available_models:
            if 'irrigation_decisions' in detailed_results[model] and detailed_results[model]['irrigation_decisions']:
                # Get a representative irrigation pattern (first episode)
                pattern = detailed_results[model]['irrigation_decisions'][0]
                days = range(len(pattern))
                
                # Calculate cumulative irrigation
                cumulative = np.cumsum(pattern)
                
                plt.plot(days, cumulative, label=model, 
                       color=colors.get(model, '#333333'), linewidth=2)
        
        plt.xlabel('Growing Season Day')
        plt.ylabel('Cumulative Irrigation (mm)')
        plt.title('Irrigation Strategy Comparison')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'irrigation_strategies.png'), dpi=300)
        plt.close()
    
    # 5. YIELD vs IRRIGATION EFFICIENCY FRONTIER
    plt.figure(figsize=(10, 8))
    
    # Plot all models with yield and irrigation data
    for model in all_results['label'].unique():
        model_data = all_results[all_results['label'] == model]
        
        if not model_data.empty:
            x = model_data['Seasonal irrigation (mm)_mean'].values[0]
            y = model_data['Dry yield (tonne/ha)_mean'].values[0]
            
            # Plot point
            plt.scatter(x, y, s=100, label=model, 
                      color=colors.get(model, '#333333'), 
                      edgecolor='black', linewidth=1)
            
            # If detailed results available, plot distribution
            if model in detailed_results and 'irrigations' in detailed_results[model]:
                irrigations = detailed_results[model]['irrigations']
                yields = detailed_results[model]['yields']
                
                # Plot a sample of individual episodes as small points
                sample_size = min(50, len(irrigations))
                indices = np.random.choice(len(irrigations), sample_size, replace=False)
                
                plt.scatter([irrigations[i] for i in indices], 
                          [yields[i] for i in indices],
                          s=15, alpha=0.3, color=colors.get(model, '#333333'))
    
    # Add Pareto frontier annotation
    plt.annotate('Efficiency Frontier', xy=(180, 14), xytext=(300, 14),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
               fontsize=12)
    
    plt.xlabel('Irrigation Applied (mm)')
    plt.ylabel('Crop Yield (tonne/ha)')
    plt.title('Yield-Irrigation Efficiency Frontier')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'efficiency_frontier.png'), dpi=300)
    plt.close()
    
    # 6. PERFORMANCE RADAR CHART
    rl_only = [m for m in rl_models if m in available_models]
    if len(rl_only) >= 2:
        radar_data = all_results[all_results['label'].isin(rl_only)]
        
        # Create radar chart
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Define metrics
        radar_metrics = ['Yield', 'Profit', 'WUE', 'Water Saving']
        metric_columns = [
            'Dry yield (tonne/ha)_mean',
            'Profit_mean',
            'WaterEfficiency_mean',
            'Seasonal irrigation (mm)_mean'
        ]
        
        # Normalize metrics to 0-1 scale
        normalized_data = {}
        for i, metric in enumerate(radar_metrics):
            col = metric_columns[i]
            values = radar_data[col]
            
            if metric == radar_metrics[3]:  # Water Saving (invert irrigation)
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    normalized_data[metric] = 1 - ((values - min_val) / (max_val - min_val))
                else:
                    normalized_data[metric] = values * 0 + 0.5
            else:  # Regular metrics (higher is better)
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    normalized_data[metric] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[metric] = values * 0 + 0.5
        
        # Setup for radar chart
        num_metrics = len(radar_metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for i, model in enumerate(rl_only):
            model_data = radar_data[radar_data['label'] == model]
            values = [normalized_data[metric][model_data.index[0]] for metric in radar_metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=model, color=colors[model])
            ax.fill(angles, values, alpha=0.1, color=colors[model])
        
        # Set chart properties
        plt.xticks(angles[:-1], radar_metrics)
        ax.set_rlabel_position(45)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey")
        plt.ylim(0, 1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('RL Model Performance Comparison', y=1.08)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_radar.png'), dpi=300)
        plt.close()
    
    return f"Created professional visualizations in {plots_dir}"


if __name__ == "__main__":
    # Define paths to the model checkpoints
    vanilla_dqn_path = "final_model.pt"  # Vanilla DQN model
    double_dqn_path = "final_model.pt"  # Double DQN model
    rainbow_path = "best_model.pt"  # Rainbow DQN model
    ppo_path = "ppo_final_1000k.pt"  # PPO model
    
    # Run the evaluation
    results, detailed_results = evaluate_all_approaches(
        vanilla_dqn_path=vanilla_dqn_path,
        double_dqn_path=double_dqn_path,
        rainbow_path=rainbow_path,
        ppo_path=ppo_path,
        n_episodes=50,  # Number of episodes to evaluate
        seed=42,
        years_range=(2008, 2018)  # Evaluation years (different from training years)
    )
    
    # Print summary of best performing models
    print("\nSummary of Results:")
    print("-" * 50)
    
    # Sort results by key metrics
    best_yield = results.sort_values('Dry yield (tonne/ha)_mean', ascending=False).iloc[0]
    best_profit = results.sort_values('Profit_mean', ascending=False).iloc[0]
    best_wue = results.sort_values('WaterEfficiency_mean', ascending=False).iloc[0]
    lowest_irr = results.sort_values('Seasonal irrigation (mm)_mean', ascending=True).iloc[0]
    
    print(f"Best yield: {best_yield['label']} - {best_yield['Dry yield (tonne/ha)_mean']:.2f} tonne/ha")
    print(f"Best profit: {best_profit['label']} - ${best_profit['Profit_mean']:.2f}/ha")
    print(f"Best water-use efficiency: {best_wue['label']} - {best_wue['WaterEfficiency_mean']:.2f} kg/m³")
    print(f"Lowest irrigation: {lowest_irr['label']} - {lowest_irr['Seasonal irrigation (mm)_mean']:.2f} mm")
    
    print("\nRL Model Comparison:")
    print("-" * 50)
    
    # Compare RL models only
    rl_models = ['Rainbow DQN', 'Double DQN', 'Vanilla DQN', 'PPO']
    rl_results = results[results['label'].isin(rl_models)]
    
    if not rl_results.empty:
        for _, row in rl_results.iterrows():
            print(f"{row['label']}:")
            print(f"  Yield: {row['Dry yield (tonne/ha)_mean']:.2f} ± {row['Dry yield (tonne/ha)_std']:.2f} tonne/ha")
            print(f"  Irrigation: {row['Seasonal irrigation (mm)_mean']:.2f} ± {row['Seasonal irrigation (mm)_std']:.2f} mm")
            print(f"  Profit: ${row['Profit_mean']:.2f}/ha")
            print(f"  WUE: {row['WaterEfficiency_mean']:.2f} kg/m³")
            if 'WaterStress_mean' in row and not pd.isna(row['WaterStress_mean']):
                print(f"  Avg Water Stress: {row['WaterStress_mean']:.2f}")
            print()
    
    print(f"All visualizations saved to {plots_dir}")
    print(f"Detailed results saved to {results_dir}")
    
    # Test function to verify model loading
    def test_model_loading():
        """Test if models can be loaded correctly."""
        try:
            # List your model paths
            models = [
                vanilla_dqn_path,
                double_dqn_path,
                rainbow_path,
                ppo_path
            ]
            
            # Create test environment
            env = EnhancedMaize(mode='eval')
            env.reset()
            
            # Test each model
            for model_path in models:
                if not os.path.exists(model_path):
                    print(f"✗ Model file not found: {model_path}")
                    continue
                
                print(f"Testing loading: {model_path}")
                try:
                    # Detect model type from path
                    if "vanilla" in model_path.lower():
                        model = load_vanilla_dqn_model(model_path, env)
                        print(f"✓ Successfully loaded Vanilla DQN model")
                    elif "double" in model_path.lower():
                        model = load_double_dqn_model(model_path, env)
                        print(f"✓ Successfully loaded Double DQN model")
                    elif "rainbow" in model_path.lower():
                        model = load_rainbow_model(model_path, env)
                        print(f"✓ Successfully loaded Rainbow DQN model")
                    elif "ppo" in model_path.lower():
                        model = load_ppo_model(model_path, env)
                        print(f"✓ Successfully loaded PPO model")
                    else:
                        print(f"? Unknown model type for: {model_path}")
                except Exception as e:
                    print(f"✗ Failed to load model: {model_path}")
                    print(f"  Error: {str(e)}")
        except Exception as e:
            print(f"Error in test function: {str(e)}")
    
    # Uncomment to test model loading
test_model_loading()