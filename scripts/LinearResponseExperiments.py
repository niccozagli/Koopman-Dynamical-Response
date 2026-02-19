import gc
import os
import pickle

import numpy as np
import psutil
from joblib import Parallel, delayed
from tqdm import tqdm

from koopman_response.systems.integrators import integrate_em
from koopman_response.systems.lorenz63 import NoisyLorenz63
from koopman_response.utils.paths import get_data_folder_path


def print_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024**2
    print(f"[{label}] Memory usage: {mem_MB:.2f} MB")


def get_observables(trajectory: np.ndarray):
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    observables = (x, y, z, x**2, y**2, z**2)
    return np.column_stack(observables)


def rho_pert(point: np.ndarray):
    x, y, z = point
    return np.array([0, x, 0])


def single_response(xi, eps, avg_obs, base_lorenz, response_integrator, seed=None):
    try:
        if seed is None:
            seed = np.random.SeedSequence().generate_state(1)[0]
        ss = np.random.SeedSequence(seed)
        rng_p, rng_m = [np.random.default_rng(s) for s in ss.spawn(2)]

        lorenzResponse = NoisyLorenz63(
            rho=base_lorenz.rho,
            sigma=base_lorenz.sigma,
            beta=base_lorenz.beta,
            noise=base_lorenz.noise,
        )

        y0_p = xi + eps * rho_pert(xi)
        _, resp_p = integrate_em(
            lorenzResponse,
            y0=y0_p,
            rng=rng_p,
            show_progress=False,
            **response_integrator,
        )

        y0_m = xi - eps * rho_pert(xi)
        _, resp_m = integrate_em(
            lorenzResponse,
            y0=y0_m,
            rng=rng_m,
            show_progress=False,
            **response_integrator,
        )

        obs_p = get_observables(resp_p) - avg_obs  # shape (T, 6)
        obs_m = get_observables(resp_m) - avg_obs

        return obs_p, obs_m
    except Exception as e:
        print(f"Error in single_response: {e}")
        return None


def main():
    # Unperturbed system
    lorenz = NoisyLorenz63(noise=2)
    base_integrator = {
        "t_span": (0, 10**5 / 2),
        "dt": 0.005,
        "tau": 100,
        "transient": 0.0,
    }
    t, X = integrate_em(lorenz, **base_integrator)
    avg_obs = get_observables(X).mean(axis=0)

    # Perturbation experiments settings
    response_integrator = {
        "t_span": (0, 30),
        "dt": 0.005,
        "tau": 1,
        "transient": 0.0,
    }

    amplitudes = [0.2, 0.4, 0.8]
    RESP_P = []
    RESP_M = []

    n_chunks = 10
    chunk_size = int(np.ceil(X.shape[0] / n_chunks))

    for eps in amplitudes:
        print_memory(f"Start eps={eps}")
        resp_p_acc = 0
        resp_m_acc = 0
        count = 0

        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            print_memory(f"  Chunk {start}-{end} before run")

            chunk_X = X[start:end]

            results = Parallel(n_jobs=-1, batch_size=10)(
                delayed(single_response)(
                    chunk_X[i], eps, avg_obs, lorenz, response_integrator
                )
                for i in range(chunk_X.shape[0])
            )

            print_memory(f"  Chunk {start}-{end} after run")

            results = [r for r in results if r is not None]
            if results:
                resp_p_chunk = np.mean([r[0] for r in results], axis=0)
                resp_m_chunk = np.mean([r[1] for r in results], axis=0)
                resp_p_acc += resp_p_chunk
                resp_m_acc += resp_m_chunk
                count += 1

            del results
            gc.collect()

        resp_p_all = resp_p_acc / count
        resp_m_all = resp_m_acc / count

        RESP_P.append(resp_p_all)
        RESP_M.append(resp_m_all)

    dictionary = {
        "Positive Response": RESP_P,
        "Negative Response": RESP_M,
        "Amplitudes": amplitudes,
        "Response Settings": response_integrator,
        "Unperturbed Settings": lorenz,
    }
    data_path = get_data_folder_path()
    f_name = "response.pkl"

    with open(data_path / f_name, "wb") as f:
        pickle.dump(dictionary, f)


if __name__ == "__main__":
    main()
