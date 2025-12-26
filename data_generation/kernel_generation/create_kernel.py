#%%
import os
import numpy as np
# from mpmath import *
# mp.dps=30000
# mp.pretty = True

# lam = [1000, 750, 500, 420, 350, 210, 200, 100]
lam = [200.0]
# lam = [210.0, 100.0, 350.0, 500.0, 750.0, 1000.0, 5000.0]
steps = 100

omega = np.linspace(-1,1,steps)
tau = np.linspace(-1,1,steps)
x_grid, y_grid = np.meshgrid(tau, omega)
x_grid = np.transpose(x_grid)
y_grid = np.transpose(y_grid)

for n in range(len(lam)):
    print(lam[n])
    K_F = np.exp(-(lam[n] / 2) * x_grid * y_grid) / (2 * np.cosh((lam[n] / 2) * tau))
    # K_F = np.nan_to_num(K_F, nan=0)
    u, s, v = np.linalg.svd(K_F)
    
    path = os.path.join('/iarai/home/daniel.springer/Projects/InvPro/kernels_100_test/', f'lambda_{int(lam[n])}')
    if not os.path.exists(path):
        os.mkdir(path)
    path_K = os.path.join(path, 'kernel.npy')
    path_u = os.path.join(path, 'u.npy')
    path_v = os.path.join(path, 'v.npy')
    path_s = os.path.join(path, 's.npy')
    np.save(path_K, K_F)
    np.save(path_u, u)
    np.save(path_v, v)
    np.save(path_s, s)


# #%%
# import matplotlib.pyplot as plt
# import numpy as np

# GA_PATH_mit = '/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/maxent/ctqmc_MIT/G_real_beta30.npy'
# C_PATH = '/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/maxent/ctqmc_MIT/c_fake_beta30.npy'
# CG_PATH =  '/iarai/home/daniel.springer/Projects/InvPro/repo/greens_function/models/maxent/ctqmc_MIT/Gc_real_beta30.npy'
# GA1 = np.load(GA_PATH_mit)

# GA_PATH_mit = '/iarai/home/daniel.springer/Projects/InvPro/new_all_model_runs/Maxent/beta30/A_maxent.npy'
# C_PATH = '/iarai/home/daniel.springer/Projects/InvPro/new_all_model_runs/Maxent/beta30/Gc.npy'
# CG_PATH =  '/iarai/home/daniel.springer/Projects/InvPro/new_all_model_runs/Maxent/beta30/Gc.npy'
# GA2 = np.load(GA_PATH_mit)

# print(GA1.shape, GA2.shape)

# plt.figure(1)
# plt.plot(GA1[0,0])
# # plt.figure(2)
# plt.plot(GA2[0,0])

# %%
