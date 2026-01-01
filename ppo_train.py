"""
Single-AUV PPO Training Script (MultiDiscrete)
NO VecNormalize
"""

import os
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from environment1auv import AUV2DEnv


# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")


# ---------------- LR schedule ----------------
def linear_schedule(initial_value: float):
    def f(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return f


# ---------------- Custom Eval Callback ----------------
class AoITrackingEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            avg_aois = []
            goal_dists = []

            for env in self.eval_env.envs:
                if hasattr(env, "AoI"):
                    avg_aois.append(float(np.mean(env.AoI)))
                if hasattr(env, "pos"):
                    goal_dists.append(float(np.linalg.norm(env.pos - env.goal)))

            if avg_aois:
                self.logger.record("eval/avg_AoI", np.mean(avg_aois))
            if goal_dists:
                self.logger.record("eval/goal_dist", np.mean(goal_dists))

        return result


# ---------------- Directories ----------------
def setup_directories():
    base = "auv2d_runs"
    models_dir = os.path.join(base, "models")
    logdir = os.path.join(base, "tb_logs")
    ckpt_dir = os.path.join(base, "checkpoints")
    best_dir = os.path.join(base, "best_model")

    for d in [models_dir, logdir, ckpt_dir, best_dir]:
        os.makedirs(d, exist_ok=True)

    return models_dir, logdir, ckpt_dir, best_dir


# ---------------- Env ----------------
def create_env(n_envs: int, monitor_dir: str):
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(
        AUV2DEnv,
        n_envs=n_envs,
        vec_env_cls=vec_cls,
        monitor_dir=monitor_dir,
    )


# ---------------- Hyperparameters ----------------
def get_hyperparameter_configs():
    return [
        {
            "name": "large_model",
            "n_steps": 512,
            "batch_size": 256,
            "learning_rate": linear_schedule(3e-4),
            "ent_coef": 0.01,
            "n_epochs": 10,
            "policy_kwargs": dict(net_arch=[256, 256]),
        },
        {
            "name": "small_model",
            "n_steps": 256,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "ent_coef": 0.02,
            "n_epochs": 8,
            "policy_kwargs": dict(net_arch=[128, 128]),
        },
    ]


# ---------------- Train ----------------
def train_one(config, models_dir, logdir, ckpt_dir, best_dir):
    run_name = f"PPO_{config['name']}"
    print("\n" + "=" * 80)
    print(f"Training: {run_name}")
    print("=" * 80)

    env = create_env(n_envs=4, monitor_dir=logdir)
    eval_env = create_env(n_envs=1, monitor_dir=logdir)

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=10_000,
            save_path=ckpt_dir,
            name_prefix=run_name,
        ),
        AoITrackingEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(best_dir, run_name),
            log_path=os.path.join(best_dir, f"{run_name}_eval"),
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    # ✅ REMOVE NON-PPO KEY ("name")
    ppo_kwargs = dict(config)
    ppo_kwargs.pop("name", None)

    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.99,
        clip_range=0.2,
        target_kl=0.03,
        verbose=1,
        tensorboard_log=logdir,
        device=device,
        **ppo_kwargs,
    )

    model.learn(
        total_timesteps=7_000_000,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=run_name,
    )

    model.save(os.path.join(models_dir, f"{run_name}_final.zip"))

    env.close()
    eval_env.close()


def main():
    models_dir, logdir, ckpt_dir, best_dir = setup_directories()

    for cfg in get_hyperparameter_configs():
        train_one(cfg, models_dir, logdir, ckpt_dir, best_dir)


if __name__ == "__main__":
    main()
