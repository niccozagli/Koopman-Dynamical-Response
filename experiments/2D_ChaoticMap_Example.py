import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.chaotic_map_2d import NoisyChaoticMap2D

    return NoisyChaoticMap2D, np, plt


@app.cell
def _():
    n_steps = 20000
    x0 = (0.1, 0.2)

    return n_steps, x0


@app.cell
def _(NoisyChaoticMap2D, n_steps, np, x0):
    system = NoisyChaoticMap2D()
    rng = np.random.default_rng(0)
    xs = system.iterate(x0=x0, n_steps=int(n_steps), rng=rng)
    return (xs,)


@app.cell
def _(plt, xs):
    fig_ts, ax_ts = plt.subplots(figsize=(7, 3))
    ax_ts.plot(xs[:, 0], lw=0.6, label="x_n")
    ax_ts.plot(xs[:, 1], lw=0.6, label="y_n")
    ax_ts.set_xlabel("n")
    ax_ts.set_ylabel("state")
    ax_ts.set_title("2D Chaotic Map Trajectory")
    ax_ts.legend()
    plt.tight_layout()
    plt.xlim((100,500))
    plt.show()
    return


@app.cell
def _(plt, xs):
    fig_map, ax_map = plt.subplots(figsize=(4, 4))
    ax_map.plot(xs[:, 0], xs[:, 1], ".", ms=1.5, alpha=0.5)
    ax_map.set_xlabel(r"$x_n$")
    ax_map.set_ylabel(r"$y_n$")
    ax_map.set_title("State Space")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
