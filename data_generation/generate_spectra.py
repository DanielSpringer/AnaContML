import argparse
import json
import numpy as np
from pathlib import Path
from typing import List
from typing import Tuple
import sys
from utils import set_permissions
from scipy.stats import skewnorm


def create_parser(args):
    parser = argparse.ArgumentParser()

    help_msg = "Path to which the generated spectra should be written to."
    parser.add_argument(
        "--outpath", type=Path, default="", help=help_msg
    )

    help_msg = "The grid of the spectrum will range from -w_max to w_max"
    parser.add_argument(
        "--w_max", type=float, help=help_msg
    )

    help_msg = "The grid will have w_steps between -w_max and w_max"
    parser.add_argument(
        "--w_steps", type=int, help=help_msg
    )

    help_msg = "Minimum and maximum value of the "
    help_msg += "width for each gaussian peak."
    parser.add_argument(
        "--sigma", type=float, nargs="+", help=help_msg
    )

    help_msg = "Maximum number of peaks within the spectrum."
    parser.add_argument(
        "--n_peaks", type=int, help=help_msg
    )

    help_msg = "Number of symmetric spectra to produce."
    parser.add_argument(
        "--n_sym_samples", type=int, help=help_msg
    )

    help_msg = "Number of asymmetric spectra to produce"
    parser.add_argument(
        "--n_asym_samples", type=int, help=help_msg
    )

    help_msg = "If the values at the boundary are above the "
    help_msg += "threshold a new sample is drawn."
    parser.add_argument(
        "--boundary_eps", type=float, default=1e-5, help=help_msg
    )

    help_msg = "Definition of the spectra generation process. "
    help_msg += "Spectra are written in a folder with this name. "
    help_msg += "The default process is `springer_spectrum`. "
    help_msg += "To implement a new process define a "
    help_msg += "member function named _{sampling_policy}_spectrum(...) which "
    help_msg += "takes `self` and `symmetric` as argument."
    parser.add_argument(
        "--sampling_policy", type=str, default="", help=help_msg
    )

    parser.add_argument(
        "--log_grid", type=str, default="False"
    )

    help_msg = "Force overwrite if folder already exists."
    parser.add_argument("--force", action="store_true", help=help_msg)

    args = parser.parse_args(args)

    return parser, args


class GenerateSpectra():

    def __init__(
        self,
        outpath: Path,
        w_max: float,
        w_steps: int,
        sigma: List[float],
        n_peaks: int,
        n_sym_samples: int,
        n_asym_samples: int,
        MIT_peak_prob: float = 0.5,
        gap_prob: float = 0.3,
        boundary_eps: float = 1e-5,
        sampling_policy: str = "springer",
        force: bool = False,
        cosine_grid: bool = False,
        log_grid: bool = False,
        log: int = 0,
        *args,
        **kwargs
    ):

        # msg = f"Upper Sigma {sigma[1]} is to large for w_max {w_max}"
        # assert w_max > 2 * sigma[1], msg
        self.w_max = w_max
        self.w_steps = w_steps
        self.sigma = sigma
        self.n_peaks = n_peaks
        self.MIT_peak_prob = MIT_peak_prob
        self.gap_prob = gap_prob

        self.n_sym_samples = int(n_sym_samples)
        self.n_asym_samples = int(n_asym_samples)
        self.boundary_eps = boundary_eps
        self.log_grid = log_grid
        self.log = log


        assert outpath.is_dir()
        assert sampling_policy, "Please specify a sampling_policy"
        self.outpath = self.prepare_outpath(outpath, sampling_policy)
        try:
            self.outpath.mkdir(parents=True, exist_ok=force)
        except FileExistsError:
            print("Found Spectra:")
            print(f'"spectra_path": "{self.outpath}",')
            raise FileExistsError
        
        self.w_grid = np.linspace(-w_max, w_max, w_steps)
        np.save(self.outpath / 'w_grid.npy', self.w_grid)

        if cosine_grid == True:
            print('Using cosine grid')
            x = np.linspace(-1,1,w_steps)
            w_neg = np.cos(0.5*np.pi*x[:int(w_steps/2)]) -1
            w_pos = np.cos(0.5*np.pi*x[int(w_steps/2):]) *(-1) +1
            w = np.concatenate([w_neg, w_pos])
            self.w_grid = w
            np.save(self.outpath / 'w_grid.npy', self.w_grid)
        if log_grid == True:
            print('Using log grid')
            x = np.linspace(1,log,int(w_steps/2))
            y = np.log(x)
            y = y/(np.max(y)) # - 1
            y = y - y[0] -1
            f = np.concatenate([y, -1*np.flip(y)])
            d = f[int(w_steps/2+1)]-f[int(w_steps/2-2)]
            f[int(w_steps/2-1)] = f[int(w_steps/2-2)] + d/3
            f[int(w_steps/2)] = f[int(w_steps/2-1)] + d/3
            self.w_grid = f
            np.save(self.outpath / 'w_grid.npy', self.w_grid)

        self.sampling_policy = sampling_policy

        if "seed" in kwargs:
            self.seed = int(kwargs["seed"])
        else:
            self.seed = 42
        np.random.seed(self.seed)
        print('SEED:', self.seed)

        self.save_configuration()

    def prepare_outpath(self, outpath: Path, sampling_policy: str) -> Path:

        w_max = str(float(self.w_max)).replace('.', 'c')
        sigma = "_".join([str(s).replace('.', 'c') for s in self.sigma])

        folders = Path(f"w_max_{w_max}_w_steps_{self.w_steps}")
        folders = folders / f"n_peaks_{self.n_peaks}_sigma_{sigma}"

        folders = folders / "spectra"
        tmp = f"{self.n_sym_samples}_sym_{self.n_asym_samples}_asym_{sampling_policy}"
        folders = folders / tmp

        return outpath / folders

    def _fournier_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''

        peaks = int(np.random.rand() * self.n_peaks)
        peaks = np.maximum(1, peaks)

        # adjust position such that peaks are between -6 to 6 for
        # w_max = 15. Otherwise, use a scaling factor to get similar
        # spectra.
        pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (6.0 * self.w_max) / 15.0
        pos[0] = (np.random.rand(1) - 0.5)

        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])

        A_norm = self._build_spectrum(pos=pos, width=width, symmetric=symmetric)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._fournier_spectrum(symmetric=symmetric)

        return A_norm

    def _springer_static_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''

        peaks = int(np.random.rand() * self.n_peaks)
        peaks = np.maximum(1, peaks)
        # adjust position such that peaks are well within w_max range
        pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])

        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])

        A_norm = self._build_spectrum(pos=pos, width=width, symmetric=symmetric)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._springer_static_spectrum(symmetric=symmetric)

        return A_norm

    # Identical as _springer_static_spectrum but with possibility of central sharp peak
    def _springer_static_2_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(width[-1,0])
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_2(pos=pos, width=width, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_2(pos=pos, width=width, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._springer_static_2_spectrum(symmetric=symmetric)

        return A_norm


    def _springer_static_3_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0], self.sigma[1]/10))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_2(pos=pos, width=width, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_2(pos=pos, width=width, symmetric=True, flag=False)
        # print('-----------------')
        # print(pos)
        # print(width)
        # print('-----------------')
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)

        return A_norm


    def _springer_static_4_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand() * 2)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0], self.sigma[1]/10))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)

        return A_norm


    def _springer_static_5_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand() * 2)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 1/4*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)

        return A_norm


    def _springer_static_6_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand() * 2)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2/3*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)

        return A_norm


    def _springer_static_7_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand() * 2)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2/3*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if np.random.rand() < self.gap_prob:
            gap = int(self.w_grid.shape[0]/2 * np.random.rand() * 0.2)
            A_norm[int(self.w_grid.shape[0]/2)-gap:int(self.w_grid.shape[0]/2)+gap] = 0
            A_norm = A_norm / (np.sum(A_norm) * (2 * self.w_max / (self.w_steps - 1)))

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm

    def _springer_static_8_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand())
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2/3*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 5)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if np.random.rand() < self.gap_prob:
            gap = int(self.w_grid.shape[0]/2 * np.random.rand() * 0.2)
            A_norm[int(self.w_grid.shape[0]/2)-gap:int(self.w_grid.shape[0]/2)+gap] = 0
            A_norm = A_norm / (np.sum(A_norm) * (2 * self.w_max / (self.w_steps - 1)))

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm


    def _springer_static_9_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand())
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 1/2*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 1/2*self.w_max)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if np.random.rand() < self.gap_prob:
            gap = int(self.w_grid.shape[0]/2 * np.random.rand() * 0.2)
            A_norm[int(self.w_grid.shape[0]/2)-gap:int(self.w_grid.shape[0]/2)+gap] = 0
            # A_norm = A_norm / (np.sum(A_norm) * (2 * self.w_max / (self.w_steps - 1)))
            A_norm = A_norm / (np.trapz(y=A_norm, x=self.w_grid))

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        # print(A_norm)
        # p = ii
        return A_norm


    def _springer_static_10_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand())
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 0.3*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 0.3*self.w_max)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if np.random.rand() < self.gap_prob:
            gap = int(self.w_grid.shape[0]/2 * np.random.rand() * 0.2)
            A_norm[int(self.w_grid.shape[0]/2)-gap:int(self.w_grid.shape[0]/2)+gap] = 0
            # A_norm = A_norm / (np.sum(A_norm) * (2 * self.w_max / (self.w_steps - 1)))
            A_norm = A_norm / (np.trapz(y=A_norm, x=self.w_grid))

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm
    

    def _springer_static_11_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand())
            # adjust position such that peaks are well within w_max range
            pos =  0.25*self.w_max + np.random.rand(peaks, 1) * 0.5*self.w_max
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos =  0.25*self.w_max + np.random.rand(peaks, 1) * 0.5*self.w_max
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if np.random.rand() < self.gap_prob:
            gap = int(self.w_grid.shape[0]/2 * np.random.rand() * 0.2)
            A_norm[int(self.w_grid.shape[0]/2)-gap:int(self.w_grid.shape[0]/2)+gap] = 0
            # A_norm = A_norm / (np.sum(A_norm) * (2 * self.w_max / (self.w_steps - 1)))
            A_norm = A_norm / (np.trapz(y=A_norm, x=self.w_grid))

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm
    

    def _springer_static_12_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = self.n_peaks
        peaks = np.maximum(1, peaks)
        # adjust position such that peaks are well within w_max range
        pos =  0.25*self.w_max + np.random.rand(peaks, 1) * 0.5*self.w_max
        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    

    def _springer_static_13_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = self.n_peaks
        peaks = np.maximum(1, peaks)
        # adjust position such that peaks are well within w_max range
        pos =  0.25*self.w_max + np.random.rand(peaks, 1) * 0.25*self.w_max
        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    

    def _springer_static_14_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = int(np.random.rand() * self.n_peaks)
        peaks = np.maximum(1, peaks)
        # adjust position such that peaks are well within w_max range
        pos =  np.random.rand(peaks, 1) * 0.25*self.w_max
        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    


    def _springer_static_15_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = np.maximum(10, int(np.random.rand() * self.n_peaks))
        peaks = np.maximum(1, peaks)
        # adjust position such that peaks are well within w_max range
        pos =  np.random.rand(peaks, 1) * 0.25*self.w_max
        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    

    def _springer_static_16_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = self.n_peaks
        # adjust position such that peaks are well within w_max range
        pos =  np.random.rand(peaks, 1) * 0.25*self.w_max
        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    

    def _springer_specific_1_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = 5 #self.n_peaks
        pos_n = []
        pos_n.append([0,0.05])
        pos_n.append([0.05,0.15])
        pos_n.append([0.05,0.15])
        pos_n.append([0.10,0.20])
        pos_n.append([0.15,0.25])
        pos = []
        for n in range(peaks):
            pos.append(pos_n[n][0] + np.random.rand() * (pos_n[n][1] - pos_n[n][0]))
        pos = np.array(pos)

        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    

    def _springer_specific_2_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = 5 #self.n_peaks
        pos_n = []
        pos_n.append([0,0.05])
        pos_n.append([0.11,0.15])
        pos_n.append([0.12,0.17])
        pos_n.append([0.16,0.23])
        pos_n.append([0.23,0.25])
        pos = []
        for n in range(peaks):
            pos.append(pos_n[n][0] + np.random.rand() * (pos_n[n][1] - pos_n[n][0]))
        pos = np.array(pos)

        width_n = []
        width_n.append([0.01,0.08])
        width_n.append([0.01,0.02])
        width_n.append([0.01,0.02])
        width_n.append([0.005,0.01])
        width_n.append([0.04, 0.055])
        width = []
        for n in range(peaks):
            width.append(width_n[n][0] + np.random.rand() * (width_n[n][1] - width_n[n][0]))
        width = np.array(width)

        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    


    def _springer_specific_3_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        peaks = int(np.random.rand() * self.n_peaks)
        peaks = max(peaks, 4)
        pos_n = []
        pos_n.append([0,0.05])
        pos_n.append([0.15,0.33])
        pos_n.append([0.05,0.33])
        pos = []
        for n in range(peaks):
            if n == 0:
                pos.append(pos_n[n][0] + np.random.rand() * (pos_n[n][1] - pos_n[n][0]))
            if n == 1:
                pos.append(pos_n[n][0] + np.random.rand() * (pos_n[n][1] - pos_n[n][0]))
            if n > 1:
                pos.append(pos_n[2][0] + np.random.rand() * (pos_n[2][1] - pos_n[2][0]))
        pos = np.array(pos)

        width_n = []
        width_n.append([0.05,0.2])
        width_n.append([0.005,0.02])
        width = []
        for n in range(peaks):
            if n < 2:
                width.append(width_n[n][0] + np.random.rand() * (width_n[n][1] - width_n[n][0]))
            if n >= 2:
                width.append(width_n[1][0] + np.random.rand() * (width_n[1][1] - width_n[1][0]))
        width = np.array(width)

        A_norm = self._build_spectrum_4(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        return A_norm    



    def _springer_static_Lorentzian_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Lorentzian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # Amplitude for the additional central peak can have larger/smaller weight (0,2) compared to symmetric peaks
            amplitude = np.ones(2*peaks+1)
            amplitude[-1] = abs(np.random.rand())
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 1/2*self.w_max)
            # print(pos.shape)
            width = np.random.rand(peaks+1, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            pos_all = []
            width_all = []
            for n in range(peaks):
                pos_all.append(pos[n,0])
                pos_all.append(-pos[n,0])
                width_all.append(width[n,0])
                width_all.append(width[n,0])
            # Central Peak
            pos_all.append(0)
            width_all.append(np.minimum(width[-1,0]/10, 2.0))
            pos = np.array(pos_all, dtype=np.float64)
            width = np.array(width_all, dtype=np.float64)
            pos = pos[:,None]
            width = width[:,None]
            A_norm = self._build_spectrum_1L(pos=pos, width=width, amplitude=amplitude, symmetric=True, flag=True)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 1/2*self.w_max)
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            A_norm = self._build_spectrum_1L(pos=pos, width=width, amplitude=None, symmetric=True, flag=False)

        if np.random.rand() < self.gap_prob:
            gap = int(self.w_grid.shape[0]/2 * np.random.rand() * 0.2)
            A_norm[int(self.w_grid.shape[0]/2)-gap:int(self.w_grid.shape[0]/2)+gap] = 0
            # A_norm = A_norm / (np.sum(A_norm) * (2 * self.w_max / (self.w_steps - 1)))
            A_norm = A_norm / (np.trapz(y=A_norm, x=self.w_grid))

        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm

        # For this data set we want reproducible spectra which may not satisfy boundary conditions.
        # else:
        #     A_norm = self._springer_static_3_spectrum(symmetric=symmetric)
        # print(A_norm)
        # p = ii
        return A_norm
        
    
    def _MIT_synthetic_pos06_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks) + 1
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = (-0.6 + np.random.rand(peaks, 1) * 0.6*2 ) * self.w_max
            pos[0] = 0
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            # width[0,0] = 0.1 * np.random.rand()
            # width[0,0] = np.maximum(width[0,0], 0.01)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = (-0.6 + np.random.rand(peaks, 1) * 0.6*2 ) * self.w_max
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])

        A_norm = self._build_spectrum(pos=pos, width=width, symmetric=symmetric)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._MIT_synthetic_pos06_spectrum(symmetric=symmetric)

        return A_norm


    
    # Identical as _springer_static_spectrum but with possibility of central sharp peak
    def _MIT_synthetic_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''
        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks) + 1
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 6 * self.sigma[1])
            # pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])
            pos[0] = 0
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            width[0,0] = 0.1 * np.random.rand()
            width[0,0] = np.maximum(width[0,0], 0.01)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 6 * self.sigma[1])
            # pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])

        A_norm = self._build_spectrum(pos=pos, width=width, symmetric=symmetric)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._MIT_synthetic_spectrum(symmetric=symmetric)

        return A_norm

    def _MIT_synthetic_center0c3_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''

        if np.random.rand() < self.MIT_peak_prob:
            # Adding 1 additional peak at the centre
            peaks = int(np.random.rand() * self.n_peaks) + 1
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])
            pos[0] = 0
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            width[0,0] = 0.3 * np.random.rand()
            width[0,0] = np.maximum(width[0,0], 0.05)
        else:
            peaks = int(np.random.rand() * self.n_peaks)
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])
            width = np.random.rand(peaks, 1)
            width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])

        A_norm = self._build_spectrum(pos=pos, width=width, symmetric=symmetric)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._MIT_synthetic_center0c3_spectrum(symmetric=symmetric)

        return A_norm

    def _springer_broadening_spectrum(self, symmetric: bool) -> np.array:
        '''Adds Gaussian peaks and norms them to get
        a single A(w) sample. Returns A[w_steps].'''

        peaks = int(np.random.rand() * self.n_peaks)
        peaks = np.maximum(1, peaks)
        # adjust position such that peaks are well within w_max range
        pos = 2 * (np.random.rand(peaks, 1) - 0.5) * (self.w_max - 2 * self.sigma[1])

        width = np.random.rand(peaks, 1)
        width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
        width = width * (1 + 2 * (abs(pos) / self.w_max))

        A_norm = self._build_spectrum(pos=pos, width=width, symmetric=symmetric)
        if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
            return A_norm
        else:
            A_norm = self._springer_broadening_spectrum(symmetric=symmetric)

        return A_norm

    def generate_gapped_spectrum(self, pos, width, skew):
        A = np.zeros(self.w_grid.shape[0])
        for a,p,s in zip(skew[:, 0], pos[:, 0], width[:, 0]):
            A += skewnorm.pdf(self.w_grid, a=a, loc=p, scale=s)
        return A

    def generate_qp(self, Z, gamma):
        A_qp = Z * gamma / (self.w_grid**2 + gamma**2)
        return A_qp
    
    def _MIT_synthetic_skewed_spectrum(self, symmetric: bool) -> np.array:
        '''
        The Hubbard bands are modeled as skewed 
        '''
        if np.random.rand() < self.MIT_peak_prob:

            # Lorentzian quasi-particle peak
            Z = np.random.rand()*2
            gamma = np.random.rand()
            A = self.generate_qp(Z=Z, gamma=gamma)

            # Draw number of Hubbard band contributions for structure
            peaks = int(np.random.rand() * self.n_peaks) + 1
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = abs((np.random.rand(peaks, 1) * 0.3 ) * self.w_max)

            ## Hubbard Bands
            width = np.maximum(self.w_max/self.w_grid.shape[0]*10, np.random.rand(peaks, 1))
            # width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            skew = abs(np.random.rand(peaks, 1)) * 10 

            left = self.generate_gapped_spectrum(pos=-pos, width=width, skew=-abs(skew))
            right = self.generate_gapped_spectrum(pos=pos, width=width, skew=abs(skew))
            
            # Combine and normalize
            A += left + right
            A_norm = A / (np.sum(A) * (2 * self.w_max / (self.w_steps - 1)))

        else:

            # Draw number of Hubbard band contributions for structure
            peaks = int(np.random.rand() * self.n_peaks) + 1
            peaks = np.maximum(1, peaks)
            # adjust position such that peaks are well within w_max range
            pos = abs((np.random.rand(peaks, 1) * 0.4 ) * self.w_max)  # Allow Hubbard bands wider outside when insulating

            ## Hubbard Bands
            width = np.maximum(self.w_max/self.w_grid.shape[0]*20, np.random.rand(peaks, 1) * 3)
            # width = self.sigma[0] + width * (self.sigma[1] - self.sigma[0])
            skew = abs(np.random.rand(peaks, 1)) * 20  # Allow for sharper gaps when insulating

            left = self.generate_gapped_spectrum(pos=-pos, width=width, skew=-abs(skew))
            right = self.generate_gapped_spectrum(pos=pos, width=width, skew=abs(skew))
            
            # Combine and normalize
            A = left + right
            A_norm = A / (np.sum(A) * (2 * self.w_max / (self.w_steps - 1)))
        
        # if A_norm[0] < self.boundary_eps and A_norm[-1] < self.boundary_eps:
        #     return A_norm
        # else:
        #     A_norm = self._MIT_synthetic_pos06_spectrum(symmetric=symmetric)
        return A_norm

    
    def _build_spectrum(
        self, pos: np.array, width: np.array, symmetric: bool) -> np.array:
        if symmetric:
            # if pos.any() > 0.4:
            #     print(pos)
            #     x = pp
            A = np.exp(-(self.w_grid - pos)**2 / width**2)
            A += np.exp(-(self.w_grid + pos)**2 / width**2)
            A = np.sum(A, axis=0)
        else:
            A = np.exp(-(self.w_grid - pos)**2 / width**2)
            A = np.sum(A, axis=0)
        # ::: Normalize spectrum to int A(w) dw = 1
        # Grid spacing: 2 * w_max / (w_steps-1) )
        A_norm = A / (np.sum(A) * (2 * self.w_max / (self.w_steps - 1)))

        return A_norm


    def _build_spectrum_2(
        self, pos: np.array, width: np.array, symmetric: bool, flag: False
    ) -> np.array:
        A = np.zeros((self.w_grid.shape[0]))
        if symmetric:
            if flag:
                for n in range(pos.shape[0]):
                    Ac = np.exp(-(self.w_grid - pos[n])**2 / width[n]**2)
                    # Ac = np.sum(Ac, axis=0)
                    A_norm = Ac / (np.sum(Ac) * (2 * self.w_max / (self.w_steps - 1)))
                    A += A_norm
            else:
                for n in range(pos.shape[0]):
                    Ac = np.exp(-(self.w_grid - pos[n])**2 / width[n]**2)
                    Ac += np.exp(-(self.w_grid + pos[n])**2 / width[n]**2)
                    # Ac = np.sum(Ac, axis=0)
                    A_norm = Ac / (np.sum(Ac) * (2 * self.w_max / (self.w_steps - 1)))
                    A += A_norm
        else:
            print('not implemented')
            exit()

        # ::: Normalize spectrum to int A(w) dw = 1
        # Grid spacing: 2 * w_max / (w_steps-1) )
        A_norm = A / (np.sum(A) * (2 * self.w_max / (self.w_steps - 1)))

        return A_norm


    def _build_spectrum_4(
        self, pos: np.array, width: np.array, amplitude: np.array, symmetric: bool, flag: False
    ) -> np.array:
        A = np.zeros((self.w_grid.shape[0]))
        if symmetric:
            if flag:
                for n in range(pos.shape[0]):
                    Ac = np.exp(-(self.w_grid - pos[n])**2 / width[n]**2)
                    # print(np.sum(Ac) * (2 * self.w_max / (self.w_steps - 1)))
                    # A_norm = amplitude[n] * Ac / (np.sum(Ac) * (2 * self.w_max / (self.w_steps - 1)))
                    A_norm = amplitude[n] * Ac / (np.trapz(y=Ac, x=self.w_grid))
                    A += A_norm
            else:
                for n in range(pos.shape[0]):
                    Ac = np.exp(-(self.w_grid - pos[n])**2 / width[n]**2)
                    Ac += np.exp(-(self.w_grid + pos[n])**2 / width[n]**2)
                    # print(np.sum(Ac))
                    if np.sum(Ac)<1e-6: print(Ac)
                    # A_norm = Ac / (np.sum(Ac) * (2 * self.w_max / (self.w_steps - 1)))
                    A_norm = Ac / (np.trapz(y=Ac, x=self.w_grid))
                    A += A_norm
        else:
            print('not implemented')
            exit()

        # ::: Normalize spectrum to int A(w) dw = 1
        # Grid spacing: 2 * w_max / (w_steps-1) )
        # A_norm = A / (np.sum(A) * (2 * self.w_max / (self.w_steps - 1)))
        A_norm = A / (np.trapz(y=A, x=self.w_grid))

        return A_norm



    def _build_spectrum_1L(
        self, pos: np.array, width: np.array, amplitude: np.array, symmetric: bool, flag: False
    ) -> np.array:
        A = np.zeros((self.w_grid.shape[0]))
        if symmetric:
            if flag:
                for n in range(pos.shape[0]):
                    Ac = 1/np.pi * ( width[n] / ( (self.w_grid - pos[n])**2 + width[n] ) )
                    A_norm = amplitude[n] * Ac / (np.trapz(y=Ac, x=self.w_grid))
                    A += A_norm
            else:
                for n in range(pos.shape[0]):
                    Ac = 1/np.pi * ( width[n] / ( (self.w_grid - pos[n])**2 + width[n] ) )
                    Ac += 1/np.pi * ( width[n] / ( (self.w_grid + pos[n])**2 + width[n] ) )
                    if np.sum(Ac)<1e-6: print(Ac)
                    A_norm = Ac / (np.trapz(y=Ac, x=self.w_grid))
                    A += A_norm
        else:
            print('not implemented')
            exit()

        # ::: Normalize spectrum to int A(w) dw = 1
        # Grid spacing: 2 * w_max / (w_steps-1) )
        # A_norm = A / (np.sum(A) * (2 * self.w_max / (self.w_steps - 1)))
        A_norm = A / (np.trapz(y=A, x=self.w_grid))

        return A_norm

    def create_spectrum(self, symmetric: bool) -> np.array:

        spectrum_settings = ["springer_static", "springer_static_2", "springer_static_3", "springer_static_4", "springer_static_5", "springer_static_6", "springer_static_7", "springer_static_8", "springer_static_9", "springer_static_10", "springer_static_11", "springer_static_12",  "springer_static_13", "springer_static_14", "springer_static_15", "springer_static_16", "springer_specific_1", "springer_specific_2", "springer_specific_3", "springer_static_Lorentzian", "springer_broadening", "fournier", "MIT_synthetic", "MIT_synthetic_center0c3", "MIT_synthetic_pos06", "MIT_synthetic_skewed"]
        if self.sampling_policy in spectrum_settings:
            spectrum_generator = getattr(self, f"_{self.sampling_policy}_spectrum")
            return spectrum_generator(symmetric)
        else:
            return self._springer_static_spectrum(symmetric)

    def generate_spectra(self, n_samples: int, symmetric: bool):
        if symmetric:
            filename = "symmetric"
        else:
            filename = "asymmetric"

        spectra = np.zeros((n_samples, self.w_steps))
        for i in range(n_samples):
            # print(i)
            # if i%5000==0: print(i, '/', n_samples)
            spectra[i] = self.create_spectrum(symmetric=symmetric)
        self.save_spectra(spectra, f"{filename}")
        del spectra

    def generate(self):

        self.generate_spectra(self.n_sym_samples, symmetric=True)
        self.generate_spectra(self.n_asym_samples, symmetric=False)

        set_permissions(self.outpath)

    def save_spectra(self, spectra: np.ndarray, outname: str):
        # ::: Save spectra into file
        np.save(self.outpath / f"{outname}", spectra)

    def save_configuration(self):
        config = {
            "n_sym_samples": self.n_sym_samples,
            "n_asym_samples": self.n_asym_samples,
            "epsilon": self.boundary_eps,
            "sigma": self.sigma,
            "w_max": self.w_max,
            "w_steps": self.w_steps,
            "n_peaks": self.n_peaks,
            "sampling_policy": self.sampling_policy,
            "seed": self.seed,
            "log_grid": self.log_grid,
            "log": self.log,
        }
        with open(self.outpath / Path(f"config.json"), "w") as f:
            json.dump(config, f, indent=4)


def main(args):
    parser, args = create_parser(args)

    GS = GenerateSpectra(
        outpath=args.outpath,
        w_max=args.w_max,
        w_steps=args.w_steps,
        n_peaks=args.n_peaks,
        sigma=args.sigma,
        n_sym_samples=args.n_sym_samples,
        n_asym_samples=args.n_asym_samples,
        boundary_eps=args.boundary_eps,
        sampling_policy=args.sampling_policy,
        force=args.force
    )
    GS.generate()


if __name__ == '__main__':
    main(sys.argv[1:])
