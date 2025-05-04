# 🌱 Advanced DQN for Smart Water Irrigation

This project applies advanced Deep Q-Network (DQN) algorithms and a Policy based algoritm to optimize water irrigation strategies in agriculture. The objective is to maximize crop yield while minimizing water usage through intelligent, data-driven decision-making.

##Overview

We implement and compare the following reinforcement learning agents:

-  Vanilla DQN  
-  Double DQN   
-  Rainbow DQN
-  Proximal Policy Optimization

Each agent is trained in a custom OpenAI Gym environment that simulates crop growth, soil moisture dynamics, and irrigation effects.

---

## ⚙ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ishita-29/advanced_dqn_water_irrigation.git
cd advanced_dqn_water_irrigation
```

### 2.Create Virtual Environment and install requirments

```bash
conda create -n irrigation_env python=3.8
conda activate irrigation_env
pip install -r requirements.txt
```


### 3. Train any agent (e.g., double_dqn.py, ppo.py):
```bash
python double_dqn.py
```
