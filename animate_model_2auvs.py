import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from stable_baselines3 import PPO

from environment2auv import AUV2DEnv2AUV  # your 2-AUV env


# ======================================================
#                  CONFIG
# ======================================================

STEP_DURATION = 25.0
# This will be overwritten from env.dock_radius in __main__
DOCKING_RADIUS = 1.0  


# ======================================================
#        RUN ONE EPISODE AND RECORD EVERYTHING
# ======================================================

def run_episode_with_model(env, model, max_steps=200, deterministic=True):
    obs = env.reset()

    traj = []
    aoi_hist = []
    data_hist = []
    coll_hist = []

    collision_count = 0

    step = 0
    done = False

    # Use docking zone from env
    dock_center = np.array(env.dock_center)
    dock_radius = getattr(env, "dock_radius", DOCKING_RADIUS)

    while not done and step < max_steps:
        traj.append(np.array(env.pos).copy())
        aoi_hist.append(env.AoI.copy())
        data_hist.append(env.data_bits)
        coll_hist.append(collision_count)

        action, _ = model.predict(obs, deterministic=deterministic)
        action = np.array(action).reshape(-1)

        obs, reward, done, info = env.step(action)

        # ---------------- COLLISION ACCOUNTING ----------------
        pos = np.array(env.pos)
        d_auv = np.linalg.norm(pos[0] - pos[1])

        d0 = np.linalg.norm(pos[0] - dock_center)
        d1 = np.linalg.norm(pos[1] - dock_center)
        inside_docking = (d0 <= dock_radius and d1 <= dock_radius)

        collided = False
        if not inside_docking:
            if isinstance(info, dict) and "collision" in info:
                collided = bool(info["collision"])
            elif hasattr(env, "d_th") and d_auv <= env.d_th:
                collided = True

        if collided:
            collision_count += 1

        step += 1

        if hasattr(env, "max_steps") and step >= env.max_steps:
            break

    # Final state
    traj.append(np.array(env.pos).copy())
    aoi_hist.append(env.AoI.copy())
    data_hist.append(env.data_bits)
    coll_hist.append(collision_count)

    return (
        np.array(traj),
        np.array(aoi_hist),
        np.array(data_hist),
        np.array(coll_hist),
    )


# ======================================================
#              ANIMATION (2 AUVs)
# ======================================================

def animate_episode(env, traj, aoi_hist, data_hist, coll_hist):

    num_steps = len(traj) - 1
    total_time_sec = num_steps * STEP_DURATION

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(env.xmin, env.xmax)
    ax.set_ylim(env.ymin, env.ymax)
    ax.set_aspect("equal")
    ax.grid(True)

    # ---- Nodes ----
    ax.scatter(env.nodes[:, 0], env.nodes[:, 1],
               c="red", marker="s", label="Nodes")

    # ---- Start positions ----
    ax.scatter(*traj[0, 0], c="green", marker="o", label="Start AUV1")
    ax.scatter(*traj[0, 1], c="lime", marker="o", label="Start AUV2")

    # ---- Docking zone ----
    dock_center = np.array(env.dock_center)
    dock_radius = getattr(env, "dock_radius", DOCKING_RADIUS)

    docking_circle = plt.Circle(
        dock_center,
        dock_radius,
        linestyle="--",
        linewidth=1.5,
        fill=False,
        color="black",
        alpha=0.8,
        label="Docking zone"
    )
    ax.add_patch(docking_circle)
    ax.scatter(dock_center[0], dock_center[1],
               c="black", marker="x", s=60, label="Dock center")

    # ---- AoI labels ----
    aoi_texts = []
    for k, (x, y) in enumerate(env.nodes):
        txt = ax.text(x + 0.15, y + 0.15,
                      f"AoI={int(aoi_hist[0, k])}",
                      fontsize=9, color="darkred")
        aoi_texts.append(txt)

    # ---- Text overlays ----
    data_text = ax.text(0.98, 0.02, "", transform=ax.transAxes,
                        fontsize=11, color="blue",
                        ha="right", va="bottom")

    coll_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                        fontsize=11, color="crimson",
                        ha="left", va="bottom")

    steps_text = ax.text(
        0.98, 0.98,
        f"Steps: {num_steps}\nTotal time: {total_time_sec:.0f} s",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.85)
    )

    # ---- Trajectories ----
    (auv1_line,) = ax.plot([], [], "b-", lw=2, label="AUV 1")
    (auv1_dot,)  = ax.plot([], [], "bo")
    (auv2_line,) = ax.plot([], [], "m-", lw=2, label="AUV 2")
    (auv2_dot,)  = ax.plot([], [], "mo")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    def update(frame):
        x1, y1 = traj[:frame+1, 0, 0], traj[:frame+1, 0, 1]
        x2, y2 = traj[:frame+1, 1, 0], traj[:frame+1, 1, 1]

        auv1_line.set_data(x1, y1)
        auv1_dot.set_data(x1[-1], y1[-1])
        auv2_line.set_data(x2, y2)
        auv2_dot.set_data(x2[-1], y2[-1])

        for k, txt in enumerate(aoi_texts):
            txt.set_text(f"AoI={int(aoi_hist[frame, k])}")

        data_kB = data_hist[frame] / 8 / 1e3
        data_text.set_text(f"Total data: {data_kB:.1f} kB")
        coll_text.set_text(f"Collisions: {int(coll_hist[frame])}")

        return (
            auv1_line, auv1_dot,
            auv2_line, auv2_dot,
            *aoi_texts, data_text, coll_text
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(traj),
        interval=300,
        blit=False,
        repeat=False
    )

    plt.show()


# ======================================================
#                       MAIN
# ======================================================

if __name__ == "__main__":

    MODEL_PATH ="auv2d_runs_2auvs_10nodes/best_model/PPO_large_model_2auv/best_model.zip"


    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = AUV2DEnv2AUV()

    # Sync global docking radius with env
    DOCKING_RADIUS = getattr(env, "dock_radius", DOCKING_RADIUS)

    model = PPO.load(MODEL_PATH, device=device)

    traj, aoi_hist, data_hist, coll_hist = run_episode_with_model(
        env, model, max_steps=200, deterministic=True
    )

    num_steps = len(traj) - 1
    total_time = num_steps * STEP_DURATION
    J_AoI = np.mean(np.mean(aoi_hist, axis=1))
    total_data_kB = data_hist[-1] / 8 / 1e3

    print("\n" + "="*60)
    print("2-AUV EPISODE SUMMARY")
    print("="*60)
    print(f"Trajectory length       : {num_steps} steps")
    print(f"Total mission time      : {total_time:.0f} s")
    print(f"Final average AoI       : {np.mean(aoi_hist[-1]):.2f}")
    print(f"Time-averaged AoI J_AoI : {J_AoI:.2f}")
    print(f"Total data sent         : {total_data_kB:.1f} kB")
    print(f"Total collisions        : {int(coll_hist[-1])}")

    animate_episode(env, traj, aoi_hist, data_hist, coll_hist)
