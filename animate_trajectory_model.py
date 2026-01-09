import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from stable_baselines3 import PPO

from environment1auv import AUV2DEnv  # your env


# ======================================================
#        RUN ONE EPISODE AND RECORD EVERYTHING
# ======================================================

def run_episode_with_model(env, model, max_steps=200, deterministic=True):
    obs = env.reset()

    traj = []
    aoi_hist = []
    data_hist = []

    total_data_bits = 0
    step = 0
    done = False

    while not done and step < max_steps:
        traj.append(env.pos.copy())
        aoi_hist.append(env.AoI.copy())
        data_hist.append(total_data_bits)

        prev_occ = env.occ.copy()

        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.array(action).reshape(-1)

        obs, reward, done, info = env.step(action)

        if not np.array_equal(prev_occ, env.occ):
            total_data_bits += 12_000 * 25  # 12 kbps × 25 s

        step += 1

        if hasattr(env, "max_steps") and step >= env.max_steps:
            break

    traj.append(env.pos.copy())
    aoi_hist.append(env.AoI.copy())
    data_hist.append(total_data_bits)

    return np.array(traj), np.array(aoi_hist), np.array(data_hist)


# ======================================================
#              ANIMATION (RUN ONCE, STATIC END)
# ======================================================

def animate_episode(env, traj, aoi_hist, data_hist, step_duration=25.0):

    num_steps = len(traj) - 1
    total_time_sec = num_steps * step_duration

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(env.xmin, env.xmax)
    ax.set_ylim(env.ymin, env.ymax)
    ax.set_aspect("equal")
    ax.grid(True)

    # ---- Static elements ----
    ax.scatter(env.nodes[:, 0], env.nodes[:, 1],
               c="red", marker="s", label="Nodes")
    ax.scatter(*env.start_pos, c="green", label="Start")
    ax.scatter(*env.goal, c="black", marker="*", s=150, label="Goal")

    # ---- AoI text per node ----
    aoi_texts = []
    for k, (x, y) in enumerate(env.nodes):
        txt = ax.text(
            x + 0.15, y + 0.15,
            f"AoI={int(aoi_hist[0, k])}",
            fontsize=9,
            color="darkred"
        )
        aoi_texts.append(txt)

    # ---- Total data counter (bottom-right) ----
    data_text = ax.text(
        0.98, 0.02,
        "Total data: 0.0 kB",
        transform=ax.transAxes,
        fontsize=11,
        color="blue",
        horizontalalignment="right",
        verticalalignment="bottom"
    )

    # ---- Steps & time (TOP-RIGHT) ----
    steps_text = ax.text(
        0.98, 0.98,
        f"Steps: {num_steps}\n"
        f"Total time: {total_time_sec:.0f} s",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
    )

    # ---- AUV trajectory ----
    (auv_line,) = ax.plot([], [], "b-", lw=2, label="AUV")
    (auv_dot,) = ax.plot([], [], "bo")

    ax.legend()

    def init():
        auv_line.set_data([], [])
        auv_dot.set_data([], [])
        return auv_line, auv_dot, *aoi_texts, data_text, steps_text

    def update(frame):
        x = traj[:frame + 1, 0]
        y = traj[:frame + 1, 1]
        auv_line.set_data(x, y)
        auv_dot.set_data([x[-1]], [y[-1]])

        for k, txt in enumerate(aoi_texts):
            txt.set_text(f"AoI={int(aoi_hist[frame, k])}")

        data_kB = data_hist[frame] / 8 / 1e3
        data_text.set_text(f"Total data: {data_kB:.1f} kB")

        return auv_line, auv_dot, *aoi_texts, data_text, steps_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(traj),
        init_func=init,
        interval=300,
        blit=False,
        repeat=False
    )

    plt.show()


# ======================================================
#                       MAIN
# ======================================================

if __name__ == "__main__":

    MODEL_PATH = "auv2d_runs_new_aoi_update/best_model/PPO_large_model/best_model.zip"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = AUV2DEnv()
    model = PPO.load(MODEL_PATH, device=device)

    print("Running PPO policy (no rendering)...")
    traj, aoi_hist, data_hist = run_episode_with_model(
        env,
        model=model,
        max_steps=200,
        deterministic=True
    )

    num_steps = len(traj) - 1
    total_time = num_steps * 25
    final_pos = traj[-1]
    final_avg_aoi = np.mean(aoi_hist[-1])

    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"Trajectory length  : {num_steps} steps")
    print(f"Total mission time : {total_time:.0f} s")
    print(f"Final AUV position : ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"Final average AoI  : {final_avg_aoi:.2f}")
    print(f"Total data sent   : {data_hist[-1] / 8 / 1e3:.1f} kB")

    print("\nAnimating episode (single run, final state stays)...")
    animate_episode(env, traj, aoi_hist, data_hist)
