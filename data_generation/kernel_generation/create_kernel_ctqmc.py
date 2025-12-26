#%%
import os
import numpy as np
# from mpmath import *
# mp.dps=30000
# mp.pretty = True

# lam = [1000, 750, 500, 420, 350, 210, 200, 100]

### DONT RUN ANY NEW KERNELS. FAT4 SETUP WAS CHANGED!!!
lam = [200.0]
### DONT RUN ANY NEW KERNELS. FAT4 SETUP WAS CHANGED!!!

steps = 1000

step_size = 2/steps
s0 = step_size/2
x = np.linspace(-1+s0,1-s0,steps)
# tau = (x+1)*40/2
# print(tau)
#%%
omega = np.linspace(-1,1,steps)
tau = x
x_grid, y_grid = np.meshgrid(tau, omega)
x_grid = np.transpose(x_grid)
y_grid = np.transpose(y_grid)

for n in range(len(lam)):
    print(lam[n])
    K_F = np.exp(-(lam[n] / 2) * x_grid * y_grid) / (2 * np.cosh((lam[n] / 2) * tau))
    # K_F = np.nan_to_num(K_F, nan=0)
    u, s, v = np.linalg.svd(K_F)
    
    path = os.path.join('/colab/Projects/work/InvPro/data/ctqmc_kernels_colab/', f'lambda_{int(lam[n])}')
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


#%%
step_size = 30/100
s0 = step_size/2
tau = np.arange(s0, 30, step_size)
# tau = np.linspace(s0,30-s0,steps)
print(tau)


# Gt30 = np.load('/iarai/home/daniel.springer/Projects/InvPro/new_all_model_runs/Maxent/beta30/G_100_real.npy')
# # Gt40 = np.load('/iarai/home/daniel.springer/Projects/InvPro/new_all_model_runs/Maxent/beta40/G_100_real.npy')
# print(Gt30[0,1,:8])
# # print(Gt30.shape, tau.shape)

# print()
# for n in range(8):
#     print(Gt30[0,1,n+1] - Gt30[0,1,n], tau[n+1] - tau[n])

# # for n in range(99):
# #     print(Gt40[0,1,n+1] - Gt40[0,1,n])
