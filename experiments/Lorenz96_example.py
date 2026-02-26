import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.lorenz96 import NoisyLorenz96
    from koopman_response.algorithms import KernelDMD , GaussianKernel
    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.utils.preprocessing import make_snapshots, standardize_global
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
        standardize_global,
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
    lorenz = NoisyLorenz96(n_state=20, forcing=8.0, noise=2.5)
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
    lags , cf = cross_correlation(momentum,momentum,dt,max_lag=1000,normalization="biased")
    plt.plot(lags,cf)
    plt.xlim((-0.2,5))
    plt.show()
    return


@app.cell
def _(X, standardize_global):
    scaled_data, mean, std = standardize_global(X)
    return (scaled_data,)


@app.cell
def _(dt, make_snapshots, np, scaled_data):
    X_snap , Y_snap, dt_eff = make_snapshots(scaled_data, stride= 10, lag=5, dt=dt)

    n_snapshots_training = 5000
    idx = np.random.choice(X_snap.shape[0], size=n_snapshots_training, replace=False)
    X_snap = X_snap[idx]
    Y_snap = Y_snap[idx]
    return X_snap, Y_snap, dt_eff


@app.cell
def _(GaussianKernel, KernelDMD, X_snap, Y_snap):
    # sigma = float(np.mean(np.linalg.norm(X_snap, axis=1)))
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
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(plt, tsvd):
    fig_sv , ax_sv = plt.subplots()
    ax_sv.plot(tsvd.S / tsvd.S[0],'.')
    ax_sv.set_yscale("log")
    plt.show()
    return


@app.cell
def _(tsvd):
    check = tsvd.U.T @ tsvd.U  

    return


@app.cell
def _(kdmd, tsvd):
    K_r, U_r, S_r = tsvd.solve_from_factorization(kdmd.A, rel_threshold=1e-4)
    return (K_r,)


@app.cell
def _(K_r, KoopmanSpectrumKDMD, dt_eff, kdmd, plt):
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(K_r,kernel=kdmd.kernel)
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
    return (spectrum,)


@app.cell
def _(np, spectrum, tsvd):
    C = tsvd.Ur * 1/np.sqrt(tsvd.Sr) @ spectrum.right_eigvecs
    return


@app.cell
def _(X_snap, kdmd, scaled_data, spectrum, tsvd):
    G_phi = spectrum.inner_product_inv_measure(
        scaled_data[::10],
        batch_size=10_000,
        kernel=kdmd.kernel,              # optional if stored in spectrum
        reference_data=X_snap,       # optional if stored
        U_r=tsvd.Ur, 
        S_r=tsvd.Sr,            # optional if stored
    )
    return


@app.cell
def _():
    # inner = (.conj().T @ f_vals) / data.shape[0]  # <phi, f>
    # coeffs = np.linalg.pinv(G_phi) @ inner
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
