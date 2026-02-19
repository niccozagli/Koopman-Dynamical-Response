import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.chaotic_map import ChaoticMap1D
    from koopman_response.algorithms.edmd import EDMD
    from koopman_response.algorithms.dictionaries import FourierDictionary
    from koopman_response.utils.preprocessing import make_snapshots 
    from koopman_response.utils import get_spectral_properties

    return (
        ChaoticMap1D,
        EDMD,
        FourierDictionary,
        get_spectral_properties,
        make_snapshots,
        np,
        plt,
    )


@app.cell
def _():
    n_steps = 100_000
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
def _(mo):
    mo.md(r"""
    ### EDMD routine
    """)
    return


@app.cell
def _(xs):
    transient = 500
    flight_time = 1
    order = 25
    data_map = xs[transient:].reshape(-1, 1)
    return data_map, flight_time, order


@app.cell
def _(
    EDMD,
    FourierDictionary,
    data_map,
    flight_time,
    make_snapshots,
    np,
    order,
):
    dictionary = FourierDictionary(order=order, dim=1, L=2 * np.pi, include_constant=True)
    edmd = EDMD(dictionary=dictionary)
    X_snap, Y_snap = make_snapshots(data_map, lag=flight_time)
    K = edmd.fit_snapshots(X_snap, Y_snap, batch_size=20000, show_progress=True)
    return (K,)


@app.cell
def _(K, get_spectral_properties):
    eigs , right_eigens , left_eigens = get_spectral_properties(K)
    return (eigs,)


@app.cell
def _(eigs, np, plt):
    fig_eigs, ax_eigs = plt.subplots(figsize=(4, 4))
    ax_eigs.plot(eigs[:20].real, eigs[:20].imag,".",color="red")
    ax_eigs.set_xlabel("Re")
    ax_eigs.set_ylabel("Im")
    ax_eigs.set_title("EDMD Eigenvalues (Fourier Dictionary)")
    thetas = np.linspace(0,2*np.pi,1000)
    ax_eigs.plot(np.cos(thetas),np.sin(thetas),'b')
    ax_eigs.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
