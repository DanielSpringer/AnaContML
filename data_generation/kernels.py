import numpy as np


def get_shinaoka_kernel(
    w_max: float,
    w_steps: int,
    beta: float,
    tau_steps: int,
    kernel_type: str = "fermionic",
    *args,
    **kwargs
) -> np.ndarray:
    '''Creates coordinate transformed Kernel [Shinaoka] and
    return basis vectors v[w_steps].'''
    x = np.linspace(-1, 1, tau_steps)
    y = np.linspace(-1, 1, w_steps)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid = np.transpose(x_grid)
    y_grid = np.transpose(y_grid)

    lam = beta * w_max

    if kernel_type == "fermionic":
        K_F = np.exp(-(lam / 2) * x_grid * y_grid) / (2 * np.cosh((lam / 2) * y))
    elif kernel_type == "bosonic":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # :: Rescaled kernel (may run into nan)
    K_F = np.nan_to_num(K_F, nan=0)
    return K_F


def get_svd(kernel, *args, **kwargs) -> np.ndarray:
    u, s, v = np.linalg.svd(kernel)

    return v

def get_svd_u(kernel, *args, **kwargs) -> np.ndarray:
    u, s, v = np.linalg.svd(kernel)

    return u
