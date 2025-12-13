#%%

WORKDIR = "/mnt/scratch/daniel/Data/AnaContML"

import h5py

import numpy as np
import multiprocessing
import concurrent.futures
from pathlib import Path
import ana_cont.continuation as cont
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Shared globals
BETA = 20
Nt = 100
tau = np.linspace(0.0, BETA, Nt)
w = np.linspace(-10., 10., num=100, endpoint=True)
w2 = np.linspace(-10., 10., num=100, endpoint=True)
noise_amplitude = 1e-3

PATH = WORKDIR+"/data_2025_new/w_max_10c0_w_steps_100/n_peaks_2_sigma_0c1_0c9/lambda_20c0/10000_sym_0_asym_MIT_synthetic_skewed/noise_level_0c001_noisy_samples_10/"

def run_maxent(im_data):
    err = np.ones_like(tau) * noise_amplitude
    model = np.ones_like(w)
    model /= np.trapz(model, w)

    probl = cont.AnalyticContinuationProblem(
        im_data=im_data.real,
        im_axis=tau,
        re_axis=w,
        kernel_mode='time_fermionic',
        beta=BETA
    )

    sol, _ = probl.solve(
        method='maxent_svd',
        alpha_determination='chi2kink',
        optimizer='newton',
        stdev=err,
        model=model
    )
    return sol.A_opt

def main():
    data = np.load(PATH + "symmetric.npy")
    N_total = int(1e4) #data.shape[0]
    im_data_list = [data[N, 0] for N in range(N_total)]

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx, max_workers=24) as executor:
        spectra = list(executor.map(run_maxent, im_data_list))

    with h5py.File("maxent_data_100.h5", "w") as f:

        f.create_dataset(f"gtau", data=data[:,0])
        f.create_dataset(f"A", data=data[:,1])
        f.create_dataset(f"w", data=np.array(w2))
        f.create_dataset(f"A_maxent", data=np.array(spectra))
        f.create_dataset(f"w_maxent", data=np.array(w))
        f.create_dataset(f"beta", data=BETA)


if __name__ == "__main__":
    main()

#%%

