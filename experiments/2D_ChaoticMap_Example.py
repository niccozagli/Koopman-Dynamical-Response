import marimo as mo

__generated_with = "0.19.11"
app = mo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.chaotic_map_2d import NoisyChaoticMap2D

    return NoisyChaoticMap2D, np, plt


@app.cell
def _():
    n_steps = 200_000
    x0 = 0.1
    y0 = 0.2
    sigma = 0.0
    seed = 0
    return n_steps, seed, sigma, x0, y0


@app.cell
def _(NoisyChaoticMap2D, n_steps, seed, sigma, x0, y0):
    system = NoisyChaoticMap2D(sigma=sigma)
    traj = system.iterate(x0=x0, y0=y0, n_steps=n_steps, seed=seed)
    return system, traj


@app.cell
def _(plt, traj):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(traj[:, 0], traj[:, 1], ".", ms=1, alpha=0.3)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$y_n$")
    ax.set_title("2D Chaotic Map Trajectory")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, traj):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(traj[:-1, 0], traj[1:, 0], ".", ms=1, alpha=0.3)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_{n+1}$")
    ax.set_title("Return Map (x)")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
