import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist

    from koopman_response.algorithms import GaussianKernel, KernelDMD
    from koopman_response import KoopmanSpectrumKDMD
    from koopman_response.utils.preprocessing import make_snapshots
    from koopman_response.algorithms.regularization import TSVDRegularizer

    return (
        GaussianKernel,
        KernelDMD,
        KoopmanSpectrumKDMD,
        Path,
        TSVDRegularizer,
        make_snapshots,
        np,
        pdist,
        plt,
        xr,
    )


@app.cell
def _(Path):
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "GSEBM"

    dataset_filename = "stochastic_warm_state_{5}.nc"
    dataset_path = data_dir / dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    n_snapshots_training = 30_000
    rng_seed = 0
    return dataset_path, n_snapshots_training, rng_seed


@app.cell
def _(dataset_path, np, rng_seed, xr):
    rng = np.random.default_rng(rng_seed)
    # Load the data
    ds = xr.open_dataset(filename_or_obj=dataset_path) 
    dt_days = float(ds.attrs["tau [days]"])
    lat_mask = ds["asymptotic_temperature"].notnull().any(dim="time")

    temperature = ds.sel(latitude=ds["latitude"][lat_mask])["asymptotic_temperature"]
    space_coord = ds.sel(latitude=ds["latitude"][lat_mask])["latitude"]
    X = temperature.values


    ds.close()
    return X, ds, dt_days, lat_mask, rng, space_coord


@app.cell
def _(X, dt_days, make_snapshots, n_snapshots_training, np, rng, space_coord):
    # Pre-processing
    # scaled_data, mean, std = standardize(X)

    ######## QUADRATURE-AWARE RESCALING #########
    # Build a normalized spatial grid in [0, 1]
    _x_grid = np.asarray(space_coord, dtype=float)

    _x_min = _x_grid.min()
    _x_max = _x_grid.max()
    if np.isclose(_x_max, _x_min):
        _x_grid = np.linspace(0.0, 1.0, X.shape[1])
    else:
        _x_grid = (_x_grid - _x_min) / (_x_max - _x_min)

    # Remove the temporal mean at each grid point
    mean_field = X.mean(axis=0)
    centered_data = X - mean_field[None, :]

    # Use trapezoidal quadrature weights so the kernel metric approximates
    # the weighted L2 norm with weight cos(pi x / 2).
    dx_weight = np.empty_like(_x_grid)
    if _x_grid.size < 2:
        dx_weight[...] = 1.0
    else:
        dx_weight[0] = 0.5 * (_x_grid[1] - _x_grid[0])
        dx_weight[-1] = 0.5 * (_x_grid[-1] - _x_grid[-2])
        if _x_grid.size > 2:
            dx_weight[1:-1] = 0.5 * (_x_grid[2:] - _x_grid[:-2])

    cos_weight = np.cos(0.5 * np.pi * _x_grid)
    kernel_weight = np.clip(cos_weight * dx_weight, a_min=0.0, a_max=None)
    scaled_data = centered_data * np.sqrt(kernel_weight)[None, :]

    #########################################
    # Snapshot data and sub-sampling
    X_snap, Y_snap, dt_eff = make_snapshots(scaled_data, dt=dt_days)
    n_train = min(n_snapshots_training, X_snap.shape[0])
    idx = rng.choice(X_snap.shape[0], size=n_train, replace=False)
    X_snap = X_snap[idx]
    Y_snap = Y_snap[idx]
    return (
        X_snap,
        Y_snap,
        centered_data,
        dt_eff,
        idx,
        kernel_weight,
        scaled_data,
    )


@app.cell
def _(
    GaussianKernel,
    KernelDMD,
    TSVDRegularizer,
    X_snap,
    Y_snap,
    idx,
    np,
    pdist,
    scaled_data,
):
    # Kernel DMD
    rel_threshold = 3e-2
    sigma_median = float(np.median(pdist(scaled_data[idx], metric="euclidean")))

    kdmd = KernelDMD(kernel=GaussianKernel(sigma=sigma_median))
    kdmd.fit_snapshots(X=X_snap, Y=Y_snap)

    tsvd = TSVDRegularizer()
    rel_threshold_svd_temporary = 1e-3
    tsvd.factorize(kdmd.G, method="eigsh", rel_threshold=rel_threshold_svd_temporary)
    return kdmd, rel_threshold_svd_temporary, tsvd


@app.cell
def _(plt, tsvd):
    fig_sv, ax_sv = plt.subplots()
    ax_sv.plot(tsvd.S / tsvd.S[0], ".")
    ax_sv.set_yscale(value="log")
    ax_sv.set_xlabel(xlabel="$i$", size=16)
    ax_sv.set_ylabel(ylabel=r"$\sigma^2_i / \sigma^2_1$", size=16)
    ax_sv.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(kdmd, np, plt, rel_threshold_svd_temporary, tsvd):
    thresholds = np.linspace(start=rel_threshold_svd_temporary, stop=0.5, num=50)
    _records = []
    for _rel_threshold in thresholds:
        _Kr, _, _ = tsvd.solve_from_factorization(
            kdmd.A,
            rel_threshold=_rel_threshold,
        )
        _records.append(_Kr.shape[0])

    _fig, _ax = plt.subplots()
    _ax.plot(thresholds, _records)
    return


@app.cell
def _(KoopmanSpectrumKDMD, dt_eff, kdmd, tsvd):
    rel_threshold_svd = 5e-3
    Kr, Ur, Sr = tsvd.solve_from_factorization(
        kdmd.A,
        rel_threshold_svd
    )
    spectrum = KoopmanSpectrumKDMD.from_koopman_matrix(
        Kr,
        kernel=kdmd.kernel,
        reference_data=kdmd.reference_data,
        U_r=Ur,
        S_r=Sr,
    )
    eigs_ct = spectrum.continuous_time_eigenvalues(dt_eff)
    return eigs_ct, spectrum


@app.cell
def _(eigs_ct, plt):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(eigs_ct.real, eigs_ct.imag, ".", ms=4)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ [day$^{-1}$]")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$ [day$^{-1}$]")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.007, right=0.001)
    ax.set_ylim(bottom=-0.001, top=0.001)
    # fig.savefig("figures/eigenvalues_near.png",dpi=400)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(eigs_ct, np):
    import marimo as mo
    lines = []
    for i in range(1, len(eigs_ct))[:10]:
        tau_years = np.abs(1 / eigs_ct[i].real) / 365
        lines.append(rf"$\tau_{{{i}}} \approx {tau_years:.2f}\,\mathrm{{years}}$")

    mo.vstack([mo.md(line) for line in lines])
    return


@app.cell
def _(scaled_data, spectrum):
    # Get the eigenfunctions evaluated on the data
    phi_vals = spectrum.evaluate_eigenfunctions(scaled_data)
    return (phi_vals,)


@app.cell
def _(X, centered_data, dt_days, np, space_coord):
    x_grid = np.asarray(space_coord, dtype=float)
    if x_grid.ndim != 1 or x_grid.shape[0] != X.shape[1]:
        x_grid = np.linspace(0.0, 1.0, X.shape[1])
    else:
        x_min = x_grid.min()
        x_max = x_grid.max()
        if np.isclose(x_max, x_min):
            x_grid = np.linspace(0.0, 1.0, X.shape[1])
        else:
            x_grid = (x_grid - x_min) / (x_max - x_min)

    weight = np.cos(0.5 * np.pi * x_grid)

    def weighted_integral(xmin, xmax,data):
        mask = (x_grid >= xmin) & (x_grid <= xmax)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Not enough grid points in interval [{xmin}, {xmax}]")
        return 0.5 * np.pi * np.trapezoid(
            weight[mask][None, :] * data[:, mask],
            x_grid[mask],
            axis=1,
        )

    global_temperature = weighted_integral(0.0, 1.0,data=centered_data)
    delta_temperature = weighted_integral(0.0, 1.0 / 3.0,data=centered_data) - weighted_integral(
        1.0 / 3.0,
        1.0,data=centered_data
    )
    time_days = dt_days * np.arange(X.shape[0])
    return delta_temperature, global_temperature


@app.cell
def _(delta_temperature, global_temperature, np, phi_vals, plt):
    from scipy.stats import binned_statistic_2d
    grid4_bins = 80
    grid4_min_count = 5
    grid4_n_eigfuncs = 4

    grid4_T = global_temperature
    grid4_dT = delta_temperature

    grid4_count, grid4_T_edges, grid4_dT_edges, _ = binned_statistic_2d(
        grid4_T,
        grid4_dT,
        None,
        statistic="count",
        bins=grid4_bins,
    )

    grid4_T_centers = 0.5 * (grid4_T_edges[:-1] + grid4_T_edges[1:])
    grid4_dT_centers = 0.5 * (grid4_dT_edges[:-1] + grid4_dT_edges[1:])

    grid4_fig, grid4_axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)
    grid4_axes = grid4_axes.ravel()

    for grid4_eig_idx, grid4_ax in enumerate(grid4_axes[:grid4_n_eigfuncs]):
        grid4_eig_idx += 1
        grid4_phi = phi_vals[:, grid4_eig_idx].real
        grid4_mean_phi, _, _, _ = binned_statistic_2d(
            grid4_T,
            grid4_dT,
            grid4_phi,
            statistic="mean",
            bins=[grid4_T_edges, grid4_dT_edges],
        )
        grid4_mean_phi_masked = np.ma.masked_where(
            grid4_count < grid4_min_count,
            grid4_mean_phi,
        )

        grid4_absmax = 0.75*np.nanmax(np.abs(grid4_mean_phi_masked))
        grid4_im = grid4_ax.pcolormesh(
            grid4_T_edges,
            grid4_dT_edges,
            grid4_mean_phi_masked.T,
            shading="auto",
            cmap="seismic",
            vmin=-grid4_absmax,
            vmax=grid4_absmax
        )

        # if np.ma.count(grid4_mean_phi_masked) > 0:
        #     grid4_levels = np.linspace(
        #         grid4_mean_phi_masked.min(),
        #         grid4_mean_phi_masked.max(),
        #         8,
        #     )
        #     if np.unique(grid4_levels).size > 1:
        #         grid4_ax.contour(
        #             grid4_T_centers,
        #             grid4_dT_centers,
        #             grid4_mean_phi_masked.T,
        #             levels=grid4_levels,
        #             colors="k",
        #             linewidths=0.8,
        #             alpha=0.9,
        #         )

        grid4_ax.set_title(rf"$\mathrm{{Re}}\,\phi_{{{grid4_eig_idx}}}$")
        grid4_ax.set_xlim(left=-5, right=5)
        grid4_ax.set_ylim(bottom=-3, top=3)
        grid4_fig.colorbar(
            grid4_im,
            ax=grid4_ax,
        )

    for grid4_ax in grid4_axes[4:]:
        grid4_ax.set_xlabel(r"$\overline{T}[K]$", size=16)
    for grid4_ax in grid4_axes[::2]:
        grid4_ax.set_ylabel(r"$\Delta T [K]$", size=16)

    for grid4_ax in grid4_axes[grid4_n_eigfuncs:]:
        grid4_ax.axis("off")

    # grid4_fig.savefig("figures/eigenfunctions_near.png",dpi=400)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(ds, lat_mask):
    warm_albedo = ds.sel(latitude=ds["latitude"][lat_mask])["warm_albedo"]
    warm_albedo_mean = warm_albedo.mean(dim="time")
    warm_albedo_std = warm_albedo.std(dim="time")
    return warm_albedo_mean, warm_albedo_std


@app.cell
def _(
    X_snap,
    kernel_weight,
    np,
    plt,
    space_coord,
    spectrum,
    tsvd,
    warm_albedo_mean,
    warm_albedo_std,
):
    warm_albedo_coord = np.asarray(warm_albedo_mean["latitude"], dtype=float)
    warm_albedo_mean_vals = np.asarray(warm_albedo_mean, dtype=float)
    warm_albedo_std_vals = np.asarray(warm_albedo_std, dtype=float)

    mode4_coord = np.asarray(space_coord, dtype=float)
    if mode4_coord.ndim != 1 or mode4_coord.shape[0] != X_snap.shape[1]:
        mode4_coord = np.arange(X_snap.shape[1], dtype=float)

    mode4_n_modes = 4
    mode4_indices = np.arange(1, mode4_n_modes + 1)
    mode4_matrix = np.column_stack(
        [spectrum.koopman_modes(observable=X_snap[:, mode4_j], U_r=tsvd.Ur, S_r=tsvd.Sr) for mode4_j in range(X_snap.shape[1])]
    ).T
    mode4_matrix = mode4_matrix / np.sqrt(kernel_weight)[:, None]

    mode4_fig, mode4_axes = plt.subplots(
        3,
        2,
        figsize=(12, 8),
        sharex=True,
    )
    mode4_axes = mode4_axes.ravel()

    albedo_axes = []
    albedo_ref_ax = None

    for mode4_ax, mode4_idx in zip(mode4_axes, mode4_indices):
        mode4_profile = mode4_matrix[:, mode4_idx]
        mode4_ax.plot(mode4_coord, mode4_profile.real, color="tab:blue")

        if albedo_ref_ax is None:
            ax2 = mode4_ax.twinx()
            albedo_ref_ax = ax2
        else:
            ax2 = mode4_ax.twinx()
            ax2.sharey(albedo_ref_ax)

        ax2.plot(warm_albedo_coord, warm_albedo_mean_vals, color="tab:red", lw=1.5,linestyle='--',alpha=0.5)
        ax2.fill_between(
            warm_albedo_coord,
            warm_albedo_mean_vals - warm_albedo_std_vals,
            warm_albedo_mean_vals + warm_albedo_std_vals,
            color="tab:red",
            alpha=0.2,
        )

        albedo_axes.append(ax2)

        mode4_ax.set_title(rf"Koopman mode for $\phi_{{{mode4_idx}}}$")
        mode4_ax.grid(alpha=0.3)

    for mode4_ax in mode4_axes[2:]:
        mode4_ax.set_xlabel(r"$x$", size=16)
    for mode4_ax in mode4_axes[::2]:
        mode4_ax.set_ylabel("Mode amplitude")

    for ax2 in albedo_axes:
        ax2.label_outer()
    for ax2 in albedo_axes[1::2]:
        ax2.set_ylabel("Warm albedo")

    # mode4_fig.savefig("figures/Koopman_modes_far.png", dpi=400)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(ds):
    ds.close()
    return


if __name__ == "__main__":
    app.run()
