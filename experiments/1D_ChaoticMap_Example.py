import marimo as mo

__generated_with = "0.19.11"
app = mo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from koopman_response.systems.chaotic_map import ChaoticMap1D
    from koopman_response.algorithms.edmd import EDMD
    from koopman_response.algorithms.dictionaries import FourierDictionary
    from koopman_response.algorithms.spectrum import KoopmanSpectrum
    from koopman_response.utils.preprocessing import make_snapshots
    from koopman_response.utils.signal import cross_correlation 

    return (
        ChaoticMap1D,
        EDMD,
        FourierDictionary,
        KoopmanSpectrum,
        cross_correlation,
        make_snapshots,
        np,
        plt,
    )


@app.cell
def _():
    n_steps = 300_000
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
def _(xs):
    transient = 500
    flight_time = 1
    order = 30
    data_map = xs[transient:].reshape(-1, 1)
    return data_map, flight_time, order


@app.cell
def _(
    EDMD,
    FourierDictionary,
    KoopmanSpectrum,
    data_map,
    flight_time,
    make_snapshots,
    np,
    order,
):
    dictionary = FourierDictionary(order=order, dim=1, L=2 * np.pi, include_constant=True)
    edmd = EDMD(dictionary=dictionary)
    X_snap, Y_snap = make_snapshots(data_map, lag=flight_time)
    K = edmd.fit_snapshots(X=X_snap, Y=Y_snap, batch_size=20000, show_progress=True)
    spectrum = KoopmanSpectrum.from_koopman_matrix(K, dictionary)
    koop_gram_matrix = spectrum.eigenfunction_inner_product(edmd.gram())
    return dictionary, edmd, koop_gram_matrix, spectrum


@app.cell
def _(np, plt, spectrum):
    eigs = spectrum.eigenvalues
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


@app.cell
def _(
    cross_correlation,
    data_map,
    flight_time,
    koop_gram_matrix,
    np,
    plt,
    spectrum,
):
    eigfunc_vals = spectrum.evaluate_eigenfunctions(data_map)
    Phi2= eigfunc_vals[:,1]

    lags , cf = cross_correlation(Phi2,Phi2,dt=flight_time,normalization="biased")
    cf_theoretical = [ koop_gram_matrix[1,1]*spectrum.eigenvalues[1]**i for i in lags ] 
    fig_corr , ax_corr = plt.subplots(nrows=2,sharex=True)
    ax_corr[0].plot(lags,np.real(cf))
    ax_corr[0].plot(lags,np.real(cf_theoretical),'o')
    ax_corr[1].plot(lags,np.imag(cf))
    ax_corr[1].plot(lags,np.imag(cf_theoretical),'o')
    ax_corr[0].set_xlim((-0.5,20))
    plt.show()
    return


@app.cell
def _(mo):
    return mo.md(
        r"""
        ### Correlation functions of observables
        """
    )


@app.cell
def _(cross_correlation, data_map, edmd, np, spectrum):
    observable = np.cos(data_map.reshape(-1))
    lags_obs , corr_obs = cf_observable = cross_correlation(observable,observable,dt=1,normalization="biased")
    psi_inner = spectrum.psi_inner(data=data_map,f_values=observable)
    f_hat =spectrum.best_coefficients(edmd.gram(),psi_inner)
    koop_corr = spectrum.correlation_function_discrete(edmd.gram(), f_hat, f_hat)
    return corr_obs, f_hat, koop_corr, lags_obs


@app.cell
def _(corr_obs, koop_corr, lags_obs, np, plt):
    fig_corr_obs , ax_corr_obs = plt.subplots()
    ax_corr_obs.plot(lags_obs,corr_obs)
    ax_corr_obs.plot(lags_obs,np.real(koop_corr(lags_obs)),'.')
    ax_corr_obs.set_xlim((-1,20))

    plt.show()
    return


@app.cell
def _(mo):
    return mo.md(
        r"""
        ### Response Function
        """
    )


@app.cell
def _(data_map, dictionary, edmd, f_hat, np, spectrum):
    X_const = np.array([1.0])  # dim=3

    delta_psi = dictionary.delta_from_constant(data_map, X_const)
    delta_gamma = spectrum.right_eigvecs.conj().T @ delta_psi
    G_phi = spectrum.eigenfunction_inner_product(edmd.gram())

    gamma = np.linalg.pinv( G_phi,rcond=1e-5) @ delta_gamma

    koop_resp = spectrum.correlation_function_discrete(G=edmd.gram(), coeff_f=f_hat, coeff_g=gamma)
    return (koop_resp,)


@app.cell
def _(koop_resp, np, plt):
    fig_response , ax_response = plt.subplots()
    t= np.arange(0,30,0.01)
    ax_response.plot(t,np.real(koop_resp(t)))
    return


if __name__ == "__main__":
    app.run()
