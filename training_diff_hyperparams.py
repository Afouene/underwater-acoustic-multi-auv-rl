"""
Multi-AUV Training Script with Hyperparameter Optimization
"""
import os
import gym
import eco2ai
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from environment2auv import MultiAUVEnvironment  # Your custom environment class


# In new_training.py, add this custom callback class
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

class AgeTrackingEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            avg_ages = []
            # Handle both single and vectorized environments
            if isinstance(self.eval_env, DummyVecEnv):
                for env in self.eval_env.envs:
                    if hasattr(env, 'reward_per_step') and len(env.reward_per_step) > 0:
                        avg_ages.append(np.mean(env.reward_per_step))
            else:
                if hasattr(self.eval_env, 'reward_per_step') and len(self.eval_env.reward_per_step) > 0:
                    avg_ages.append(np.mean(self.eval_env.reward_per_step))
            
            if avg_ages:
                self.logger.record('eval/avg_age', np.mean(avg_ages))
        
        return result
    


    

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")


def setup_directories():
    """Create required directory structure"""
    models_dir = "7_nodes_2_auvs_models"
    logdir = "7_nodes_2_auvs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    return models_dir, logdir


def create_environment(n_envs=8):
    """Create vectorized environment"""
    return make_vec_env(
        MultiAUVEnvironment,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        env_kwargs={},
        monitor_dir="7_nodes_2_auvs"
    )


def get_hyperparameter_configs():
    return [
        {
            'name': 'large_model',
            'n_steps':       512,
            'batch_size':    1024,
            'gamma':         0.99,
            ' learning_rate': lambda frac: 3e-4 * frac,
            'ent_coef':      0.01,
            'clip_range':    0.2,
            'n_epochs':      16,
            'policy_kwargs': dict(net_arch=[256,256]),
            'target_kl':     0.02,
        },
        {
            'name': 'second_model',
            'n_steps': 256,
            'batch_size': 256,
            'gamma': 0.097,
            'learning_rate': 1e-4,
            'ent_coef': 0.02,
            'clip_range': 0.2,
            'n_epochs': 12,
            'policy_kwargs': dict(net_arch=[128, 128]),
            'target_kl': 0.03
        }
    ]


def create_callbacks():
    """Create training callbacks"""
    # Create separate evaluation environment
    eval_env = DummyVecEnv([lambda: MultiAUVEnvironment()])
    
    checkpoint = CheckpointCallback(
        save_freq=5000,
        save_path="./7_nodes_2_auvs/checkpoints",
        name_prefix="ppo_auv"
    )
    
    evaluator = AgeTrackingEvalCallback(
        eval_env,
        best_model_save_path="./7_nodes_2_auvs/best_model",
        log_path="./7_nodes_2_auvs/results",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    return CallbackList([checkpoint, evaluator])

def train_model(config, env, models_dir, logdir):
    """Train a single model configuration"""
    print(f"\nTraining configuration: {config['name']}")
    print("Hyperparameters:")
    for k, v in config.items():
        if k != 'policy_kwargs':
            print(f"{k}: {v}")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        learning_rate=config['learning_rate'],
        ent_coef=config['ent_coef'],
        clip_range=config['clip_range'],
        n_epochs=config['n_epochs'],
        policy_kwargs=config['policy_kwargs'],
        verbose=1,
        tensorboard_log=logdir,
        device=device
    )

    model.learn(
        total_timesteps=15_000_000,
        callback=create_callbacks(),  # Remove env parameter
            progress_bar=True,
            tb_log_name=f"PPO_{config['name']}"
        )
    
    model.save(f"{models_dir}/PPO_{config['name']}_final")
    
    return model

def main():
    """Main training workflow"""
    models_dir, logdir = setup_directories()
    env = create_environment(n_envs=8)
    
    for config in get_hyperparameter_configs():
        try:
            model = train_model(config, env, models_dir, logdir)
            del model  # Clear memory between experiments
        except Exception as e:
            print(f"Error training {config['name']}: {str(e)}")
            continue

    env.close()

if __name__ == "__main__":
    main()
