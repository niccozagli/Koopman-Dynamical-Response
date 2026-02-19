import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.chaotic_map import ChaoticMap1D

    return ChaoticMap1D, plt


@app.cell
def _():
    n_steps = 10000
    x0 = 0.1
    return n_steps, x0


@app.cell
def _(ChaoticMap1D, n_steps, x0):
    system = ChaoticMap1D()
    xs = system.iterate(x0=float(x0), n_steps=int(n_steps))
    return (xs,)


@app.cell
def _(plt, xs):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, lw=0.6)
    ax.set_xlabel("n")
    ax.set_ylabel("x_n")
    ax.set_title("1D Chaotic Map Trajectory")
    plt.tight_layout()
    plt.xlim(0,200)
    plt.show()
    return


@app.cell
def _(plt, xs):
    fig_hist , ax_hist = plt.subplots()
    ax_hist.hist(xs,bins=100)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
