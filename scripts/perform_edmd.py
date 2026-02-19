import pickle

from koopman_response.algorithms.dictionaries import ChebyshevDictionary
from koopman_response.algorithms.edmd import EDMD
from koopman_response.systems.integrators import integrate_em
from koopman_response.systems.lorenz63 import NoisyLorenz63
from koopman_response.utils.preprocessing import make_snapshots, normalise_data_chebyshev
from koopman_response.utils.paths import get_data_folder_path

######### CHOOSE THE PARAMETERS FOR EDMD ##########
degrees = [13, 14, 15, 16, 17, 18, 19, 20]
flight_times = [1]

# Integrate the Lorenz system
lorenz = NoisyLorenz63()
t, X = integrate_em(
    lorenz,
    t_span=(0.0, 10**6),
    dt=0.001,
    tau=100,
    transient=500.0,
    show_progress=True,
)
# Scale the data
scaled_data, data_min, data_max = normalise_data_chebyshev(X)

list_degree = []
for degree in degrees:
    list_ftime = []
    for f_time in flight_times:
        dictionary = ChebyshevDictionary(degree=degree, dim=3)
        edmd = EDMD(dictionary=dictionary)
        X_snap, Y_snap = make_snapshots(scaled_data, lag=f_time)
        K = edmd.fit_snapshots(X_snap, Y_snap)
        list_ftime.append(edmd)
    list_degree.append(list_ftime)

results = {"edmd results": list_degree, "lorenz settings": lorenz}

data_path = get_data_folder_path()
f_name = "edmd_smalldt.pkl"
with open(data_path / f_name, "wb") as f:
    pickle.dump(results, f)
