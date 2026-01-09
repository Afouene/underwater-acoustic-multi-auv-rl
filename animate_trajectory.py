import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================================================
#            IMPORT ENV + BASELINE (UNCHANGED)
# ======================================================

from environment1auv import AUV2DEnv
from baseline_greedy_aoi_goal import greedy_aoi_goal_policy
from baseline_communication_first import communication_first_policy
from baseline_round_robin import round_robin_policy


# ======================================================
#        RUN ONE EPISODE AND RECORD EVERYTHING
# ======================================================

def run_episode(env, policy_fn, max_steps=200):
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

        action = policy_fn(env)
        obs, reward, done, _ = env.step(action)

        if not np.array_equal(prev_occ, env.occ):
            total_data_bits += 12_000 * 25  # 12 kbps × 25 s

        step += 1

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

    # ---- Total data (bottom-right) ----
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
        auv_dot.set_data(x[-1], y[-1])

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
        interval=80,
        blit=False,
        repeat=False
    )

    plt.show()


# ======================================================
#                       MAIN
# ======================================================

if __name__ == "__main__":

    env = AUV2DEnv()

    print("Running greedy baseline (no rendering)...")
    traj, aoi_hist, data_hist = run_episode(
        env,
        policy_fn=greedy_aoi_goal_policy,
        max_steps=200
    )

    num_steps = len(traj) - 1
    total_time = num_steps * 25
    final_avg_aoi = np.mean(aoi_hist[-1])

    print(f"Final average AoI   : {final_avg_aoi:.2f}")
    print(f"Trajectory length  : {num_steps} steps")
    print(f"Total mission time : {total_time:.0f} s")

    print("Animating episode (single run, final state stays)...")
    animate_episode(env, traj, aoi_hist, data_hist)
