"""
Two-AUV PPO Training Script (MultiDiscrete)
- NO VecNormalize
- Stable-Baselines3 PPO
- Custom eval logs: avg AoI, avg goal distance (only for AUVs not reached), avg inter-AUV distance,
  collision_rate (d_auv < d_th), avg collision penalty (from env params)
"""

import os
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from environment2auv import AUV2DEnv2AUV   # <-- change if your file name differs


# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")


# ---------------- LR schedule ----------------
def linear_schedule(initial_value: float):
    def f(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return f


# ---------------- Custom Eval Callback ----------------
class MultiAUVTrackingEvalCallback(EvalCallback):
    """
    Logs extra evaluation metrics directly from the eval env state.
    Assumes env has:
      - AoI (shape N,)
      - pos (shape (2,2))
      - goal (shape (2,))
      - reached (shape (2,), bool)
      - d_th (float, meters)
    """
    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            avg_aois = []
            goal_dists = []
            d_auv_list = []
            col_events = []
            col_penalties = []

            for env in self.eval_env.envs:
                # unwrap Monitor/VecEnv wrappers when possible
                e = getattr(env, "unwrapped", env)

                if hasattr(e, "AoI"):
                    avg_aois.append(float(np.mean(e.AoI)))

                # goal distance: average over AUVs that have NOT reached yet (more informative)
                if hasattr(e, "pos") and hasattr(e, "goal") and hasattr(e, "reached"):
                    pos = np.array(e.pos)
                    goal = np.array(e.goal)
                    reached = np.array(e.reached, dtype=bool)

                    if pos.shape[0] >= 2:
                        dists = [float(np.linalg.norm(pos[i] - goal)) for i in range(pos.shape[0]) if not reached[i]]
                        if len(dists) > 0:
                            goal_dists.append(float(np.mean(dists)))
                        else:
                            goal_dists.append(0.0)

                # inter-AUV distance + collision stats
                if hasattr(e, "pos") and np.array(e.pos).shape[0] >= 2:
                    d = float(100.0 * np.linalg.norm(np.array(e.pos)[0] - np.array(e.pos)[1]))  # meters
                    d_auv_list.append(d)

                    if hasattr(e, "d_th"):
                        col_events.append(1.0 if d < float(e.d_th) else 0.0)

                    # approximate collision penalty using same formula (thresholded risk)
                    if hasattr(e, "d_th") and hasattr(e, "sigma_c") and hasattr(e, "alpha_c"):
                        violation = max(0.0, float(e.d_th) - d)
                        risk = 1.0 - np.exp(-(violation**2) / (2.0 * float(e.sigma_c)**2))
                        penalty = -float(e.alpha_c) * risk
                        col_penalties.append(penalty)

            if avg_aois:
                self.logger.record("eval/avg_AoI", float(np.mean(avg_aois)))
            if goal_dists:
                self.logger.record("eval/avg_goal_dist_unreached", float(np.mean(goal_dists)))
            if d_auv_list:
                self.logger.record("eval/avg_d_auv_m", float(np.mean(d_auv_list)))
                self.logger.record("eval/min_d_auv_m", float(np.min(d_auv_list)))
            if col_events:
                self.logger.record("eval/collision_rate", float(np.mean(col_events)))
            if col_penalties:
                self.logger.record("eval/avg_collision_penalty", float(np.mean(col_penalties)))

        return result


# ---------------- Directories ----------------
def setup_directories():
    base = "auv2d_runs_2auvs_13nodes"
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
        AUV2DEnv2AUV,
        n_envs=n_envs,
        vec_env_cls=vec_cls,
        monitor_dir=monitor_dir,
    )


# ---------------- Hyperparameters ----------------
def get_hyperparameter_configs():
    # Multi-AUV is harder; slightly higher entropy + a bit larger net often helps
    return [
        {
            "name": "large_model_2auv",
            "n_steps": 512,
            "batch_size": 256,
            "learning_rate": linear_schedule(3e-4),
            "ent_coef": 0.02,
            "n_epochs": 10,
            "policy_kwargs": dict(net_arch=[256, 256]),
        },
        {
            "name": "small_model_2auv",
            "n_steps": 1024,
            "batch_size": 256,
            "learning_rate": linear_schedule(3e-4),
            "ent_coef": 0.02,
            "n_epochs": 10,
            "policy_kwargs": dict(net_arch=[256, 256]),
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
        MultiAUVTrackingEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(best_dir, run_name),
            log_path=os.path.join(best_dir, f"{run_name}_eval"),
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    # remove non-PPO key
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
        total_timesteps=10_000_000,
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
