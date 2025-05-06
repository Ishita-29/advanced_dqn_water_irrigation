# 🌱 Advanced DQN and PPO for Smart Water Irrigation

This project applies advanced Deep Q-Network (DQN) algorithms and a Policy-based algorithm to optimize water irrigation strategies in agriculture. The objective is to maximize crop yield while minimizing water usage through intelligent, data-driven decision-making.

This project builds upon CleanRL, a high-quality collection of single-file reinforcement learning implementations. Modified versions of selected agents (e.g., DQN, PPO) were adapted to train in the custom irrigation environment.

Link to CleanRL github: https://github.com/vwxyzjn/cleanrl

## Overview

We implement and compare the following reinforcement learning agents:

- Vanilla DQN  
- Double DQN   
- Rainbow DQN
- Proximal Policy Optimization (PPO)

Each agent is trained in a custom OpenAI Gym environment that simulates crop growth, soil moisture dynamics, and irrigation effects.

---

## ⚙ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ishita-29/advanced_dqn_water_irrigation.git
cd advanced_dqn_water_irrigation
```

### 2. Create Virtual Environment and Install Requirements

```bash
conda create -n irrigation_env python=3.8
conda activate irrigation_env
pip install -r requirements.txt
```

### 3. Train Any Agent

```bash
# Train Double DQN
python double_dqn_clean_rl_enhanced_500k.py

# Train Rainbow DQN
python rainbow_clean_rl_enhanced_500k.py

# Train Vanilla DQN
python vanilladqn_clean_rl_enhanced_env_500k.py

# Train PPO
python ppo_clean_rl_enhanced_500k.py
```

## 🌾 Custom Environment: EnhancedMaize

The EnhancedMaize environment is a reinforcement learning environment built on AquaCrop-OSPy that simulates maize crop growth and irrigation management. This environment allows RL agents to learn optimal irrigation strategies that balance maximizing crop yield with minimizing water usage under various climate conditions.

### Key Features

- **Detailed Observation Space**: 29-dimensional state vector containing comprehensive crop metrics (age, canopy cover, biomass), soil water conditions (depletion, total available water), growth stage indicators, water stress metrics, and a 7-day weather forecast.

- **Granular Action Space**: Five discrete irrigation options (0mm, 5mm, 10mm, 15mm, 25mm) offering more control compared to binary irrigation decisions.

- **Enhanced Reward Structure**: Sophisticated reward mechanism that balances immediate water conservation with long-term yield maximization, accounting for growth stage criticality.

- **Water Stress Tracking**: Explicit modeling of plant water stress to guide decision-making, with penalties proportional to stress level, especially during critical growth stages.

- **Historical Weather Data**: Support for training on historical climate data (1982-2007) with evaluation on different years (2008-2018) to test generalization.

### Environment Design

The environment follows the standard Gymnasium interface with:

```python
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
action_space = spaces.Discrete(5)  # [0mm, 5mm, 10mm, 15mm, 25mm]
```

#### Observation Space Breakdown

Each observation includes:

1. **Crop Parameters**:
   - Age in days
   - Canopy cover percentage
   - Biomass growth
   - Soil water depletion
   - Total available water (TAW)
   - Normalized growth stage (0-1)
   - Water stress indicator (0-1, where 1 is maximum stress)
   - Current water use efficiency

2. **Weather Data**:
   - Daily precipitation forecast (7 days)
   - Daily minimum temperature forecast (7 days)
   - Daily maximum temperature forecast (7 days)

#### Reward Mechanism

The reward system is designed to address the sparse-action, delayed-reward nature of agricultural irrigation:

1. **Step Reward**: For each daily decision
   - For irrigation actions:
     - Positive reward when irrigation is applied during high water stress
     - Higher rewards during critical growth stages (flowering/grain filling)
     - Penalties for unnecessary irrigation when water stress is low
   - For non-irrigation actions:
     - Positive reward when water stress is low (conservation is good)
     - Penalties when skipping irrigation during high stress
     - Higher penalties during critical growth stages

2. **Terminal Reward**: At harvest time
   - Based on final dry yield
   - Additional bonuses for water efficiency
   - Combines yield and efficiency into a comprehensive success metric

## 📊 Evaluation Framework

The repository includes a comprehensive evaluation framework in `evaluation_final.py` for comparing different irrigation strategies:

- **Traditional strategies**: Rainfed, threshold-based, interval-based, and net irrigation
- **RL-based strategies**: Various DQN variants and PPO
- **Baseline strategies**: Random policy and simple rule-based approaches

### Performance Metrics

- **Crop Yield** (tonne/ha): Final maize yield at the end of the season
- **Total Irrigation** (mm): Volume of irrigation water applied
- **Water Efficiency** (kg/ha/mm): Yield produced per millimeter of irrigation
- **Profitability** ($): Net economic gain, factoring in crop yield and irrigation costs

### Running Evaluation

```bash
python evaluation_final.py
```

This will generate comprehensive visualizations in the `eval_output_1000k_final/plots` directory, comparing all approaches across multiple metrics.

## 🔄 Agents

### Vanilla DQN

Basic implementation of Deep Q-Network with experience replay and target networks.

### Double DQN

Extends Vanilla DQN with a dueling architecture and double Q-learning to reduce overestimation bias.

### Rainbow DQN

A comprehensive DQN variant featuring:
- Noisy Networks for exploration
- Dueling architecture
- Distributional RL (Categorical DQN)
- Prioritized Experience Replay
- N-step returns
- Double Q-learning

### Proximal Policy Optimization (PPO)

A policy gradient method that directly optimizes the policy using clipped surrogate objectives.

## 📈 Results

Our experimental results demonstrate that:

1. Rainbow DQN achieves the highest water efficiency while maintaining competitive yield
2. Double DQN provides a good balance between yield and water conservation
3. PPO shows strong adaptability to varying climate conditions
4. All RL methods significantly outperform traditional irrigation strategies in terms of water efficiency

Detailed visualization comparisons can be found in the directory after running evaluation.

## 📁 Project Structure

```
├── aquacropgymnasium/
│   ├── env.py                          
├── double_dqn.py # Double DQN implementation
├── evaluation_final.py                  
├── ppo.py        
├── rainbow_dqn.py    
├── vanilladqn.py                  
└── README.md                          
```
