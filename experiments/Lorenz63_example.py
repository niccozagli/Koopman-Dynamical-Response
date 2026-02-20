import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from koopman_response import KoopmanSpectrum
    from koopman_response.algorithms.edmd import EDMD
    from koopman_response.algorithms.dictionaries import ChebyshevDictionary
    from koopman_response.algorithms.regularization import TSVDRegularizer
    from koopman_response.systems.lorenz63 import NoisyLorenz63
    from koopman_response.utils.signal import cross_correlation
    from koopman_response.utils.preprocessing import make_snapshots, minmax_scale

    return (
        ChebyshevDictionary,
        EDMD,
        KoopmanSpectrum,
        NoisyLorenz63,
        TSVDRegularizer,
        cross_correlation,
        make_snapshots,
        minmax_scale,
        np,
        plt,
    )


@app.cell
def _(NoisyLorenz63):
    lorenz = NoisyLorenz63()
    t, X = lorenz.integrate_em_jit(
        t_span=(0.0, 10_000),
        dt=0.001,
        tau=10,
        transient=1000.0,
    )
    dt = t[1] - t[0]
    return X, dt


@app.cell
def _(X, cross_correlation, dt, plt):
    fig_cf ,ax_cf =plt.subplots()
    signal = X[:,2]
    lags, cf = cross_correlation(signal,signal,dt=dt,normalization="biased")

    ax_cf.plot(lags,cf)
    plt.xlim((-0.5,15))
    plt.show()
    return


@app.cell
def _(ChebyshevDictionary, EDMD, X, dt, make_snapshots, minmax_scale):
    scaled_data, data_min, data_max = minmax_scale(data=X, feature_range=(-1.0, 1.0))
    lag = 10
    dt_eff = dt * lag
    X_snap, Y_snap = make_snapshots(scaled_data, lag=lag)

    dictionary = ChebyshevDictionary(degree=15, dim=3)
    edmd = EDMD(dictionary=dictionary, dt_eff=dt_eff)
    K = edmd.fit_snapshots(X_snap, Y_snap, batch_size=10000, show_progress=True)
    return dictionary, edmd, scaled_data


@app.cell
def _(KoopmanSpectrum, TSVDRegularizer, dictionary, edmd):
    tsvd = TSVDRegularizer(rel_threshold=1e-4)
    K_r, U_r, S_r = tsvd.solve(edmd.G, edmd.A)

    spectrum = KoopmanSpectrum.from_koopman_matrix(K_r,dictionary)
    eigs_ct = spectrum.continuous_time_eigenvalues(dt_eff=edmd.dt_eff)
    return S_r, U_r, eigs_ct, spectrum


@app.cell
def _(eigs_ct, plt):
    fig_eigs, ax_eigs = plt.subplots(figsize=(4, 4))
    ax_eigs.plot(eigs_ct.real, eigs_ct.imag, "x", ms=6)
    ax_eigs.set_xlabel("Re")
    ax_eigs.set_ylabel("Im")
    ax_eigs.grid(alpha=0.4)
    ax_eigs.set_title("Eigenvalues Generator")
    plt.xlim(-1.5,0.2)
    plt.ylim(-20,20)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(S_r, U_r, np, scaled_data, spectrum):
    observable = scaled_data[:,2]
    psi_inner = spectrum.psi_inner(scaled_data, observable )   # <psi,f>
    c_hat = (np.diag(1/np.sqrt(S_r)) @ U_r.conj().T) @ psi_inner
    f_hat = spectrum.left_eigvecs.conj().T @ c_hat
    return f_hat, observable


@app.cell
def _(cross_correlation, dt, eigs_ct, f_hat, np, observable, spectrum):
    G_hat = np.eye(f_hat.shape[0], dtype=np.complex128)
    koop_corr = spectrum.correlation_function_continuous(
        G=G_hat,
        coeff_f=f_hat,
        coeff_g=f_hat,
        eigenvalues=eigs_ct,
    )
    lags_obs, corr_obs = cross_correlation(observable,observable,dt=dt,normalization="biased",max_lag=10**4)
    return corr_obs, koop_corr, lags_obs


@app.cell
def _(corr_obs, koop_corr, lags_obs, np, plt):
    fig_corr_obs, ax_corr_obs = plt.subplots()
    ax_corr_obs.plot(lags_obs, corr_obs)
    ax_corr_obs.plot(lags_obs, np.real(koop_corr(lags_obs)), ".",ms=3)
    ax_corr_obs.set_xlim((-1, 15))
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
