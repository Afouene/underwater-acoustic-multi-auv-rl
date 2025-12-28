import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================================================
#            IMPORT ENV + BASELINE (UNCHANGED)
# ======================================================

from environment1auv import AUV2DEnv          # your env
from baseline_greedy_aoi_goal import greedy_aoi_goal_policy     # your baseline


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
        # record state BEFORE action
        traj.append(env.pos.copy())
        aoi_hist.append(env.AoI.copy())
        data_hist.append(total_data_bits)

        prev_occ = env.occ.copy()

        action = policy_fn(env)
        obs, reward, done, _ = env.step(action)

        # check if a successful transmission happened
        if not np.array_equal(prev_occ, env.occ):
            # rate = 12 kbps, duration = 25 s
            total_data_bits += 12_000 * 25

        step += 1
    traj.append(env.pos.copy())
    aoi_hist.append(env.AoI.copy())
    data_hist.append(total_data_bits)

    return (
        np.array(traj),
        np.array(aoi_hist),
        np.array(data_hist),
    )


# ======================================================
#              ANIMATION (RUN ONCE, STATIC END)
# ======================================================

def animate_episode(env, traj, aoi_hist, data_hist):
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

    # ---- Total data counter ----
    data_text = ax.text(
    0.98, 0.02,
    "Total data: 0.0 kB",
    transform=ax.transAxes,
    fontsize=11,
    color="blue",
    horizontalalignment="right",
    verticalalignment="bottom"
)


    # ---- AUV trajectory ----
    (auv_line,) = ax.plot([], [], "b-", lw=2, label="AUV")
    (auv_dot,) = ax.plot([], [], "bo")

    ax.legend()

    def init():
        auv_line.set_data([], [])
        auv_dot.set_data([], [])
        return auv_line, auv_dot, *aoi_texts, data_text

    def update(frame):
        # trajectory
        x = traj[:frame + 1, 0]
        y = traj[:frame + 1, 1]
        auv_line.set_data(x, y)
        auv_dot.set_data(x[-1], y[-1])

        # AoI update
        for k, txt in enumerate(aoi_texts):
            txt.set_text(f"AoI={int(aoi_hist[frame, k])}")

        # data counter (bits → kB)
        data_kB = data_hist[frame] / 8 / 1e3
        data_text.set_text(f"Total data: {data_kB:.1f} kB")

        return auv_line, auv_dot, *aoi_texts, data_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(traj),
        init_func=init,
        interval=80,
        blit=False,
        repeat=False      # IMPORTANT: run once only
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

    print("Animating episode (single run, final state stays)...")
    animate_episode(env, traj, aoi_hist, data_hist)
