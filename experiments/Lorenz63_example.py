import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.integrators import integrate_em
    from koopman_response.systems.lorenz63 import NoisyLorenz63
    from koopman_response.utils.signal import cross_correlation

    return NoisyLorenz63, cross_correlation, integrate_em, np, plt


@app.cell
def _():
    t_max = 5000
    dt = 0.001
    tau = 10
    noise = 2
    return dt, noise, t_max, tau


@app.cell
def _(NoisyLorenz63, dt, integrate_em, noise, np, t_max, tau):
    lorenz = NoisyLorenz63(noise=float(noise))

    rng = np.random.default_rng()
    t, X = integrate_em(
        lorenz,
        t_span=(0.0, float(t_max)),
        dt=float(dt),
        tau=int(tau),
        transient=0.0,
        rng=rng,
        show_progress=True,
    )
    return (X,)


@app.cell
def _(X, plt):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.7, alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Noisy Lorenz 63 Trajectory")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X, cross_correlation, dt):
    x,y,z = X[:,0] , X[:,1] , X[:,2]
    records_cf = []
    for signal in [x,y,z]:
        l , c = cross_correlation(signal,signal,dt=dt)
        records_cf.append((l,c))
    return (records_cf,)


@app.cell
def _(plt, records_cf):
    fig_cf ,ax_cf =plt.subplots(nrows=3,sharex=True)
    for i, (lags, correlation_function)  in enumerate(records_cf):
        ax_cf[i].plot(lags,correlation_function)

    plt.xlim((0,5))
    ax_cf[0].set_ylim((0,70))
    ax_cf[1].set_ylim((0,70))
    ax_cf[2].set_ylim((-30,30))
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
