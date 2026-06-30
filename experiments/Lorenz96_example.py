import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.lorenz96 import NoisyLorenz96
    from koopman_response.algorithms import KernelDMD , GaussianKernel
    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.utils.preprocessing import make_snapshots, standardize
    from koopman_response.algorithms.regularization import TSVDRegularizer
    from koopman_response.utils import cross_correlation

    return (
        GaussianKernel,
        KernelDMD,
        KoopmanSpectrumKDMD,
        NoisyLorenz96,
        TSVDRegularizer,
        cross_correlation,
        make_snapshots,
        mo,
        np,
        plt,
        standardize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Response properties to perturbations of Lorenz96

    The Lorenz96 model is an $N$-dimensional chaotic system with cyclic indices and
    additive noise. The equations are
    $$
    dx_i = \big( (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F \big)\,dt + \eta\, dW_i,
    $$
    with $i=1,\dots,N$ and $x_{i+N} = x_i$.
    """)
    return


@app.cell
def _(NoisyLorenz96):
    lorenz = NoisyLorenz96(n_state=20, forcing=8.0, noise=1.5)
    t, X = lorenz.integrate_em_jit(
        t_span=(0.0, 10_000.0),
        dt=0.001,
        tau=10,
        transient=1000
    )
    dt = float(t[1] - t[0])

    momentum = X.mean(axis=1)
    energy = (X**2).mean(axis=1)
    return X, dt, momentum


@app.cell
def _(cross_correlation, dt, momentum, plt):
    lags , cf = cross_correlation(
        x  = momentum,
        y  = momentum,
        dt = dt,max_lag=1000,
        normalization="biased")
    plt.plot(lags,cf)
    # plt.xlim((-0.2,5))
    plt.show()
    return cf, lags


@app.cell
def _(X, standardize):
    scaled_data, mean, std = standardize(X)
    return scaled_data, std


@app.cell
def _(dt, make_snapshots, np, scaled_data):
    X_snap , Y_snap, dt_eff = make_snapshots(scaled_data, stride= 10, dt=dt)

    n_snapshots_training = 10_000
    idx = np.random.choice(X_snap.shape[0], size=n_snapshots_training, replace=False)
    X_snap = X_snap[idx]
    Y_snap = Y_snap[idx]
    return X_snap, Y_snap, dt_eff


@app.cell
def _(np, scaled_data):
    from scipy.spatial.distance import pdist

    m = 10_000  # subsample size
    idx_sub = np.random.choice(scaled_data.shape[0], size=min(m, scaled_data.shape[0]), replace=False)
    sigma = np.median(pdist(scaled_data[idx_sub], metric="euclidean"))
    print(sigma)
    return


@app.cell
def _(GaussianKernel, KernelDMD, X_snap, Y_snap):
    kdmd = KernelDMD(kernel=GaussianKernel(sigma=6))
    _ = kdmd.fit_snapshots(X=X_snap,Y=Y_snap)
    return (kdmd,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Factorisation of Gram matrix in RKHS
    """)
    return


@app.cell
def _(TSVDRegularizer, kdmd):
    tsvd = TSVDRegularizer()
    _ = tsvd.factorize(kdmd.G, method="eigh")
    return (tsvd,)


@app.cell
def _(plt, tsvd):
    fig_sv , ax_sv = plt.subplots()

    ax_sv.plot(tsvd.S/tsvd.S[0],'.')
    ax_sv.set_yscale("log")
    ax_sv.set_xlabel("$i$",size=16)
    ax_sv.set_ylabel(r"$\sigma^2_i / \sigma^2_1$",size=16)
    # ax_sv.set_xlim((-10,1000))
    plt.show()
    return


@app.cell
def _(kdmd, tsvd):
    Kr, Ur, Sr = tsvd.solve_from_factorization(
        kdmd.A,
        rel_threshold=1e-5
        )
    print(Kr.shape)
    return (Kr,)


@app.cell
def _(KoopmanSpectrumKDMD, Kr, dt_eff, kdmd, plt):
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(Kr,kernel=kdmd.kernel)
    eigs_ct = spectrum.continuous_time_eigenvalues(dt_eff)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(eigs_ct.real, eigs_ct.imag, ".", ms=4)
    ax.set_xlabel(r"$\mathrm{Re} \lambda$")
    ax.set_ylabel(r"$\mathrm{Im} \lambda$")
    ax.set_title("KDMD Spectrum (Reduced Koopman)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.xlim((-3,0.1))
    plt.show()
    return eigs_ct, spectrum


@app.cell
def _(X_snap, spectrum, tsvd):
    observable = X_snap.mean(axis=1)
    koopman_modes = spectrum.koopman_modes(
        observable,
        U_r=tsvd.Ur,
        S_r=tsvd.Sr,
    )
    return (koopman_modes,)


@app.cell
def _(X_snap, cf, eigs_ct, koopman_modes, lags, plt, spectrum, std, tsvd):
    scalar_product_phi = spectrum.eigenfunction_gram(
        S_r=tsvd.Sr,
        n_samples=X_snap.shape[0],
        normalize=True,
    )

    cf_kdmd = spectrum.correlation_function_continuous(
        G_phi=scalar_product_phi,
        coeff_f=koopman_modes,
        coeff_g=koopman_modes,
        eigenvalues=eigs_ct
    )

    fig_final, ax_final = plt.subplots()
    ax_final.plot(lags, cf , label="Time Series")
    ax_final.set_xlim((-0.1,5))
    ax_final.plot(lags,cf_kdmd(lags).real  *  std[2]**2,label="KDMD reconstruction")
    ax_final.set_xlabel(r"$t$",size=16)
    ax_final.set_ylabel(r"$t$",size=16)
    ax_final.legend()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
