import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from aquacrop.core import AquaCropModel
from aquacrop.entities.crop import Crop
from aquacrop.entities.inititalWaterContent import InitialWaterContent
from aquacrop.entities.irrigationManagement import IrrigationManagement
from aquacrop.entities.soil import Soil
from aquacrop.utils import prepare_weather

class EnhancedMaize(gym.Env):
    def __init__(
        self,
        render_mode=None,
        mode='train',
        year1=1982,
        year2=2018,
        crop='Maize',
        climate_file=None,
        planting_date=None
    ):
        super(EnhancedMaize, self).__init__()
        self.year1 = year1
        self.year2 = year2
        self.init_wc = InitialWaterContent(value=['FC'])
        self.crop_name = crop
        self.climate = climate_file if climate_file is not None else 'champion_climate.txt'

        base_path = os.path.dirname(os.path.dirname(__file__))
        self.climate_file_path = os.path.abspath(os.path.join(base_path, 'weather_data', self.climate))

        self.planting_date = planting_date if planting_date is not None else '05/01'
        self.soil = Soil('SandyLoam')

        self.crop_growth_cycles = {
            'Maize': 130,
            'Wheat': 150,
            'Rice': 120,
            'Potato': 140,
        }

        # Enhanced observation space to include more detailed crop and water metrics
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)

        # More granular irrigation options
        self.action_depths = [0, 5, 10, 15, 25]
        self.action_space = spaces.Discrete(len(self.action_depths))
        
        # Track previous water stress for reward calculation
        self.prev_water_stress = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        sim_year = np.random.randint(self.year1, self.year2 + 1)
        self.simcalyear = sim_year

        self.crop = Crop(self.crop_name, self.planting_date)
        self.crop_growth_cycle = self.crop_growth_cycles.get(self.crop_name, 130)# Store for later use

        try:
            self.wdf = prepare_weather(self.climate_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Climate file not found at {self.climate_file_path}")

        self.wdf['Year'] = self.simcalyear
        self.total_irrigation_applied = 0
        self.cumulative_reward = 0
        self.prev_water_stress = 0
        
        # Additional tracking variables for enhanced reward calculation
        self.daily_biomass = []
        self.irrigation_events = []

        self.model = AquaCropModel(
            f'{self.simcalyear}/{self.planting_date}',
            f'{self.simcalyear}/12/31',
            self.wdf,
            self.soil,
            self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=5),
            initial_water_content=self.init_wc
        )

        self.model.run_model()
        
        # Store initial biomass
        self.daily_biomass.append(self.model._init_cond.biomass)

        obs = self._get_obs()
        info = {'dry_yield': 0.0, 'total_irrigation': 0, 'water_use_efficiency': 0.0}
        return obs, info

    def _get_obs(self):
        cond = self.model._init_cond
        precip_last_7_days = self._get_last_7_days_values('Precipitation')
        min_temp_last_7_days = self._get_last_7_days_values('MinTemp')
        max_temp_last_7_days = self._get_last_7_days_values('MaxTemp')
        weather_obs = np.concatenate([precip_last_7_days, min_temp_last_7_days, max_temp_last_7_days])

        # Calculate growth stage (normalized to 0-1)
        growth_stage = min(1.0, cond.age_days / self.crop_growth_cycle)
        
        # Calculate water stress indicator (0-1, where 1 is maximum stress)
        water_stress = min(1.0, max(0.0, cond.depletion / cond.taw)) if cond.taw > 0 else 1.0
        
        # Calculate water use efficiency so far
        wue = 0.0
        if self.total_irrigation_applied > 0:
            wue = cond.biomass / self.total_irrigation_applied
        
        # Enhanced observation including growth stage and water stress
        enhanced_obs = np.array([
            cond.age_days,
            cond.canopy_cover,
            cond.biomass,
            cond.depletion,
            cond.taw,
            growth_stage,
            water_stress,
            wue,
        ], dtype=np.float32)

        obs = np.concatenate([enhanced_obs, weather_obs])
        return obs

    def _get_last_7_days_values(self, column):
        current_day = self.model._clock_struct.time_step_counter
        last_7_days = self.wdf.iloc[max(0, current_day - 7):current_day][column]
        if len(last_7_days) < 7:
            padding = np.zeros(7 - len(last_7_days))
            last_7_days = np.concatenate([padding, last_7_days])
        return last_7_days

    def step(self, action):
        depth = self.action_depths[int(action)]
        self.model._param_struct.IrrMngt.depth = depth
        self.model.run_model(initialize_model=False)
        
        next_obs = self._get_obs()
        terminated = self.model._clock_struct.model_is_finished
        self.total_irrigation_applied += depth
        
        # Get current conditions for reward calculation
        cond = self.model._init_cond
        
        # Track daily biomass
        self.daily_biomass.append(cond.biomass)
        
        # Calculate biomass growth since last step
        biomass_growth = 0
        if len(self.daily_biomass) >= 2:
            biomass_growth = self.daily_biomass[-1] - self.daily_biomass[-2]
        
        # Calculate water stress (0-1, where 1 is maximum stress)
        water_stress = min(1.0, max(0.0, cond.depletion / cond.taw)) if cond.taw > 0 else 1.0
        
        # Track irrigation events
        if depth > 0:
            self.irrigation_events.append((cond.age_days, depth, water_stress))
        
        # Enhanced reward calculation based on water stress and irrigation decision
        growth_stage = min(1.0, cond.age_days / self.crop_growth_cycle)
        
        # Identify critical growth stages (flowering/grain filling)
        is_critical_stage = 0.3 < growth_stage < 0.7
        
        # Base reward calculation
        if depth > 0:
            # Reward irrigation when needed, penalize when not needed
            if water_stress > 0.6:
                # Higher reward during critical stages
                stage_factor = 1.5 if is_critical_stage else 1.0
                step_reward = (5 * water_stress * stage_factor) - (depth * 0.2)
            else:
                # Penalize unnecessary irrigation
                step_reward = -depth * (1.0 - water_stress)
        else:
            # Reward conservation when appropriate, penalize when plant needs water
            if water_stress < 0.5:
                step_reward = 1.0
            else:
                # Stronger penalty during critical stages
                stage_factor = 1.5 if is_critical_stage else 1.0
                step_reward = -water_stress * 4 * stage_factor
        
        # Add biomass growth component to reward
        step_reward += biomass_growth * 0.5
        
        # Update previous water stress
        self.prev_water_stress = water_stress
        
        self.cumulative_reward += step_reward

        if terminated:
            try:
                # Get final stats with proper error handling
                final_stats = self.model._outputs.final_stats
                
                # Check for different possible keys for yield
                if 'Dry yield (tonne/ha)' in final_stats.columns:
                    dry_yield = final_stats['Dry yield (tonne/ha)'].mean()
                elif 'Yield (tonne/ha)' in final_stats.columns:
                    dry_yield = final_stats['Yield (tonne/ha)'].mean()
                elif 'Yield' in final_stats.columns:
                    dry_yield = final_stats['Yield'].mean()
                else:
                    print("Warning: Could not find yield stats. Using biomass estimate.")
                    # Estimate yield from final biomass
                    dry_yield = self.model._init_cond.biomass * 0.01
                
                # Check for irrigation keys
                if 'Seasonal irrigation (mm)' in final_stats.columns:
                    total_irrigation = final_stats['Seasonal irrigation (mm)'].mean()
                elif 'Irrigation (mm)' in final_stats.columns:
                    total_irrigation = final_stats['Irrigation (mm)'].mean()
                else:
                    # Use our tracked value instead
                    total_irrigation = self.total_irrigation_applied
                
            except Exception as e:
                print(f"Error in terminal state processing: {e}")
                # Fallback to estimates
                dry_yield = self.model._init_cond.biomass * 0.01
                total_irrigation = self.total_irrigation_applied
            
            # Calculate water use efficiency
            wue = 0 if total_irrigation == 0 else dry_yield / total_irrigation
            
            # Enhanced terminal reward that balances yield and efficiency
            yield_reward = (dry_yield ** 3) + (wue * 50)
            
            self.cumulative_reward += yield_reward
            info = {
                'dry_yield': dry_yield, 
                'total_irrigation': total_irrigation,
                'water_use_efficiency': wue,
                'irrigation_events': self.irrigation_events
            }
            total_reward = self.cumulative_reward
            self.cumulative_reward = 0
        else:
            info = {
                'dry_yield': 0.0, 
                'total_irrigation': self.total_irrigation_applied,
                'current_biomass': cond.biomass,
                'water_stress': water_stress
            }
            total_reward = step_reward
            
        return next_obs, total_reward, terminated, False, info

    def close(self):
        pass