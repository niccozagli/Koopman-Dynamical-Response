import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
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
        mo,
        np,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Response properties to perturbations of Lorenz63

    In this notebook, we evaluate in a pure data-driven way how perturbations affect the chaotic Lorenz 63 model.


    In particular, we consider the following equations
    $$
    \begin{aligned}
    dx &= \sigma (y - x)\,dt + \eta \, dW_x,\\
    dy &= \big(x(\rho(t) - z) - y\big)\,dt + \eta \, dW_y,\\
    dz &= \big(xy - \beta z\big)\,dt + \eta \, dW_z.
    \end{aligned}
    $$

    The deterministic part is the classical Lorenz63 model with standard parameters $\sigma = 10$, $\rho_0 = 28$, and $ \beta = \frac{8}{3}$. There is also a noisy component characterised by independent Wiener processes $W_{i}$ with $i \in {x,y,z}$ and strength $\eta$.

    We study the effect of changes in the control parameter $\rho(t) = \rho_0 + \varepsilon T(t)$ where $T(t)$ is a generic time modulation of the perturbation and $\varepsilon$ is the amplitude of the perturbation. The response of any observable $f$ of the system can be written in a linear response regime as $R_f(t) = (G \star T)(t)$, where $G_f$ is the Green's function of the system associated to the observable $z$.

    In this notebook, we perform a Koopman analysis of the unperturbed timeseries (when $\varepsilon=0$) to extract relevant decorrelation/sensitivity timescales of the system. In particular, the Koopman analysis allows us to write the Green's function as

    $$
    G_f(t) \approx \sum_k G_k \, e^{\lambda_k t}.
    $$
    Each term corresponds to a **Koopman eigenfunction** with eigenvalue $\lambda_k$:
    - $\mathrm{Re}(\lambda_k)$ gives the **decay rate** of that mode,
    - $\mathrm{Im}(\lambda_k)$ gives its **oscillation frequency**.


    The Koopman analysis allows for estimating both $\lambda_k$ and $G_k$ from data.

    ---
    First, we simulate the dynamics to get the timeseries.
    """)
    return


@app.cell
def _(NoisyLorenz63):
    lorenz = NoisyLorenz63()
    t, X = lorenz.integrate_em_jit(
        t_span=(0.0, 10_000),
        dt=0.001,
        tau=10,
        transient=1000.0,
    )
    dt = float( t[1] - t[0] )
    return X, dt, t


@app.cell
def _(X, cross_correlation, dt, plt, t):
    fig_cf ,ax_cf =plt.subplots(ncols=2,figsize=(16,6))

    signal = X[:,2]
    lags, cf = cross_correlation(signal,signal,dt=dt,normalization="biased")

    ax_cf[0].plot(t,signal)
    ax_cf[1].plot(lags,cf)

    ax_cf[0].set_xlim((5500,5550))
    ax_cf[0].set_xlabel(r"$t$",size=20)
    ax_cf[1].set_xlabel(r"$t$",size=20)
    ax_cf[0].set_ylabel(r"$z(t)$",size=20)
    ax_cf[1].set_ylabel(r"$C_z(t)$",size=20)
    ax_cf[1].set_xlim((-0.5,15))
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### **Koopman analysis of the timeseries**
    We perform a Koopman analysis of the timeseries to identify the relevant correlation timescales of the system.
    For an observable $f(x)$, the autocorrelation
    $$
    C_f(t) = \langle f, U^t f \rangle_{\rho_0}
    $$
    measures how long the system fluctuations of $f$ remain correlated along its chaotic trajectory. The Koopman operator $U^t$ propagates observables forward in time, so this correlation can be decomposed into a sum of modes:
    $$
    C_f(t) \approx \sum_k f_k \, e^{\lambda_k t}.
    $$
    Each term corresponds to a **Koopman eigenfunction** with eigenvalue $\lambda_k$:
    - $\mathrm{Re}(\lambda_k)$ gives the **decay rate** of that mode,
    - $\mathrm{Im}(\lambda_k)$ gives its **oscillation frequency**.

    Physically, this means the correlation function is a superposition of decaying oscillations,
    and the Koopman spectrum provides the intrinsic timescales that control how fast
    correlations vanish.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### Extended Dynamic Mode Decomposition
    To estimate the properties of the Koopman operator, we use the Extended Dynamic Mode Decomposition (EDMD) algorithm.

    The starting point is the multivariate time series
    $\{x(t_n)\}_{n=0}^{N}$ with $x(t_n)\in\mathbb{R}^3$, sampled at a fixed time step $\Delta t = t_{n+1}-t_n$.
    The goal of EDMD is to construct, in a fully data-driven way, a finite-dimensional approximation of the Koopman operator associated with this continuous-time system.

    ### 1. Data preprocessing

    Since we will use **tensorised Chebyshev polynomials** as dictionary functions, we first rescale the data componentwise to the interval $[-1,1]$. We denote $\tilde x(t_n)$ the rescaled time series.

    ### 2. Time-series snapshot pairs

    From the time series we construct data pairs
    $$
    \tilde x_n = \tilde x(t_n),
    \quad
    \tilde y_n = \tilde x(t_{n+1}),
    \quad n=0,\dots,M-1.
    $$

    These pairs approximate the action of the flow map $\Phi^{\Delta t}$ over one time step,
    $
    \tilde y_n \approx \Phi^{\Delta t}(\tilde x_n).
    $
    """)
    return


@app.cell
def _(X, dt, make_snapshots, minmax_scale):
    scaled_data, data_min, data_max = minmax_scale(
        data=X,
        feature_range=(-1.0, 1.0)
    )

    lag = 10
    dt_eff = dt * lag
    X_snap, Y_snap = make_snapshots(scaled_data, lag=lag)
    return X_snap, Y_snap, dt_eff, scaled_data


@app.cell
def _(mo):
    mo.md(r"""
    ### 3. Dictionary of observables

    We choose a finite dictionary of observables
    $$
    \psi(\tilde x) = \big(\psi_1(\tilde x),\dots,\psi_N(\tilde x)\big),
    $$
    where the functions $\psi_k$ are **tensorised Chebyshev polynomials** up to a prescribed maximal degree.

    These functions span the approximation space
    $
    \mathcal V = \mathrm{span}\{\psi_1,\dots,\psi_N\}.
    $

    ### 4. EDMD matrices

    Using time averages along the trajectory, we define the empirical matrices
    $$
    G
    =
    \frac{1}{M}\sum_{n=0}^{M-1}
    \psi(\tilde x_n)^{*}\,\psi(\tilde x_n),
    $$

    $$
    A
    =
    \frac{1}{M}\sum_{n=0}^{M-1}
    \psi(\tilde x_n)^{*}\,\psi(\tilde y_n).
    $$

    For long trajectories, these matrices approximate
    $
    G \approx \langle \psi,\psi\rangle_{\rho_0},
    \quad
    A \approx \langle \psi, U^{\Delta t}\psi\rangle_{\rho_0},
    $
    where $\rho_0$ is the invariant measure of the unperturbed system and $U^{\Delta t}$ is the Koopman operator.
    """)
    return


@app.cell
def _(ChebyshevDictionary, EDMD, X_snap, Y_snap, dt_eff):
    dictionary = ChebyshevDictionary(degree=15, dim=3)
    edmd = EDMD(dictionary=dictionary, dt_eff=dt_eff)
    K = edmd.fit_snapshots(X_snap, Y_snap, batch_size=20000, show_progress=True)
    return dictionary, edmd


@app.cell
def _(mo):
    mo.md(r"""
    ### 5. Orthonormalisation and truncation

    To improve numerical stability, we perform a singular value decomposition of the Gram matrix
    $$
    G = U\,\Sigma\,U^\dagger.
    $$
    We retain only the $r$ largest singular values, obtaining $U_r$ and $\Sigma_r$, and define the orthonormalised dictionary
    $
    \hat{\psi}(\tilde x) = \psi(\tilde x) U_r  \Sigma_r^{-1/2}.
    $
    ### 6. Reduced Koopman operator

    In the orthonormal basis $\hat\psi$, the projected Koopman operator is
    $$
    \hat K
    =
    \Sigma_r^{-1/2}
    U_r^\dagger
    A
    U_r
    \Sigma_r^{-1/2}.
    $$

    This matrix represents the action of the Koopman operator on the truncated approximation space.
    """)
    return


@app.cell
def _(TSVDRegularizer, edmd):
    tsvd = TSVDRegularizer(rel_threshold=1e-4)
    K_r, U_r, S_r = tsvd.solve(edmd.G, edmd.A)
    return K_r, S_r, U_r


@app.cell
def _(mo):
    mo.md(r"""
    ### 7. Koopman eigenvalues and eigenfunctions

    We solve the eigenvalue problem
    $$
    \hat K \hat\xi_k = \mu_k \hat\xi_k.
    $$

    The corresponding Koopman eigenfunctions are
    $$
    \phi_k(\tilde x) = \hat\psi(\tilde x)\,\hat\xi_k.
    $$

    Since the system is continuous in time, the associated generator eigenvalues are obtained as
    $$
    \lambda_k = \frac{1}{\Delta t}\log(\mu_k).
    $$
    """)
    return


@app.cell
def _(K_r, KoopmanSpectrum, dictionary, edmd, plt):
    spectrum = KoopmanSpectrum.from_koopman_matrix(K_r,dictionary)
    eigs_ct = spectrum.continuous_time_eigenvalues(dt_eff=edmd.dt_eff)

    fig_eigs, ax_eigs = plt.subplots(figsize=(4, 4))
    ax_eigs.plot(eigs_ct.real, eigs_ct.imag, "x", ms=6)
    ax_eigs.set_xlabel(r"$\mathbf{Re}\lambda_k$",size=16)
    ax_eigs.set_ylabel(r"$\mathbf{Im}\lambda_k$",size=16)
    ax_eigs.grid(alpha=0.4)
    ax_eigs.set_title("Eigenvalues Koopman Generator")
    plt.xlim(-1.5,0.2)
    plt.ylim(-20,20)
    plt.tight_layout()
    plt.show()
    return eigs_ct, spectrum


@app.cell
def _(mo):
    mo.md(r"""
    ### 8. Projection of observables onto Koopman modes

    Given an observable $f(\tilde x)$, we first project it onto the orthonormal basis:
    $$
    \hat c
    =
    \langle \hat\psi,f\rangle_{\rho_0}
    \approx
    \frac{1}{M}\sum_{n=0}^{M-1}
    \hat\psi(\tilde x_n)^*\,f(\tilde x_n).
    $$

    We then express $f$ in terms of Koopman eigenfunctions
    $$
    f(\tilde x) \approx \sum_{k=1}^r a_k\,\phi_k(\tilde x).
    $$ by solving
    $
    a = \hat\Xi^{+}\hat c,
    $
    where $\hat\Xi = (\hat\xi_1,\dots,\hat\xi_r)$ is the matrix of right eigenvectors of the reduced Koopman matrix.
    """)
    return


@app.cell
def _(S_r, U_r, np, scaled_data, spectrum):
    observable = scaled_data[:,2]
    psi_inner = spectrum.psi_inner(scaled_data, observable )   
    c_hat = (np.diag(1/np.sqrt(S_r)) @ U_r.conj().T) @ psi_inner
    f_hat = spectrum.left_eigvecs.conj().T @ c_hat
    return f_hat, observable


@app.cell
def _(mo):
    mo.md(r"""
    ### 9. Koopman decomposition of the correlation function

    Once the observable is expanded in the Koopman eigenfunctions,
    $$
    f(\tilde x) \approx \sum_k a_k \, \phi_k(\tilde x),
    $$
    the autocorrelation function can be written as
    $$
    C_f(t)
    =
    \langle f, U^t f \rangle_{\rho_0}
    =
    \sum_{k,\ell}
    \bar a_\ell \, a_k \,
    \langle \phi_\ell, \phi_k \rangle_{\rho_0}\,
    e^{\lambda_k t}.
    $$
    """)
    return


@app.cell
def _(cross_correlation, dt, eigs_ct, f_hat, np, observable, plt, spectrum):
    G_hat = np.eye(f_hat.shape[0], dtype=np.complex128)
    koop_corr = spectrum.correlation_function_continuous(
        G=G_hat,
        coeff_f=f_hat,
        coeff_g=f_hat,
        eigenvalues=eigs_ct,
    )
    lags_obs, corr_obs = cross_correlation(observable,observable,dt=dt,normalization="biased",max_lag=10**4)

    fig_corr_obs, ax_corr_obs = plt.subplots()
    ax_corr_obs.plot(lags_obs, corr_obs,label="Time Series")
    ax_corr_obs.plot(lags_obs, np.real(koop_corr(lags_obs)), ".",ms=3,label="Koopman Reconstruction")
    ax_corr_obs.set_xlim((-1, 15))
    ax_corr_obs.set_xlabel(r"$t$",size=16)
    ax_corr_obs.set_ylabel(r"$C_z(t)$",size=16)
    ax_corr_obs.legend()
    plt.show()
    return G_hat, lags_obs


@app.cell
def _(mo):
    mo.md(r"""
    ### Response functions

    The Green's function $G_f$ can be written, using the fluctuation dissipation theorem, as a correlation function
    $
    G_f(t) = C_{f\Gamma}(t)
    $
    between the observable $f$ and the response observable $\Gamma = \frac{-\nabla\cdot (\mathbf{X}(\mathbf{x} \rho_0))}{\rho_0}$ where $\mathbf{X}(\mathbf{x})$ is the applied perturbation in the phase space. Perturbation of the parameter $\rho$ can be written using the perturbation $\mathbf{X}(\mathbf{x}) = (0,x,0)$.

    It is possible to find the decomposition of the response observable onto the orthonormal basis $\hat{\psi}$ by first evaluating the coefficients
    $$
    \Delta_i =
    \frac{1}{M}\sum_{n=0}^{M-1}
    X(x_n)\cdot\nabla\psi_i^*(x_n).
    $$

    We project $\Delta$ onto the orthonormalised basis and then onto Koopman modes:
    $$
    \hat\Delta = \Sigma_r^{-1/2} U_r^\dagger \Delta,
    \qquad
    \gamma = \Xi^\dagger \hat\Delta.
    $$

    The approximation of the response observable onto Koopman eigenfunctions is
    $$
    \Gamma(x) = \sum_k \gamma_k \phi_k(x)
    $$
    """)
    return


@app.cell
def _(
    G_hat,
    S_r,
    U_r,
    dictionary,
    eigs_ct,
    f_hat,
    lags_obs,
    np,
    plt,
    scaled_data,
    spectrum,
):
    # Only dimension 1 (y) is active for this perturbation
    perturbation ={
        1 : scaled_data[:,0]
    }
    # Only dimension 1 is active
    delta = dictionary.delta_from_trajectory(
        data=scaled_data,
        X_values=perturbation
    )

    c_hat_gamma = (np.diag(1/np.sqrt(S_r)) @ U_r.conj().T) @ delta
    gamma = spectrum.left_eigvecs.conj().T @ c_hat_gamma


    koop_resp = spectrum.correlation_function_continuous(
        G=G_hat,
        coeff_f=f_hat,
        coeff_g=gamma,
        eigenvalues=eigs_ct,
    )

    fig_resp, ax_resp = plt.subplots()
    ax_resp.plot(lags_obs, koop_resp(lags_obs).real)
    ax_resp.set_xlim((-1, 15))
    ax_resp.set_xlabel(r"$t$",size=16)
    ax_resp.set_ylabel(r"$G_f(t)$",size=16)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
