import argparse
import json
import numpy as np
from pathlib import Path
from kernels import get_shinaoka_kernel, get_svd, get_svd_u
import sys
from utils import set_permissions
from typing import List


def create_parser(args):
    parser = argparse.ArgumentParser()

    help_msg = "Path to the directory generated for one specific `w_max` "
    help_msg += "in the previous step. E.g `Path('_max_5/spectra/test')`. "
    help_msg += "Note that the path must be given as a Path object"
    parser.add_argument(
        "--spectra_path", type=Path, default="", help=help_msg
    )

    help_msg = "Beta value which is used in combination with `w_max` "
    help_msg += "to calculate the corresponding Kernel. Note that "
    help_msg += "`tau_steps` are chosen to be the same as `w_steps`"
    parser.add_argument(
        "--beta", type=float, nargs="+", help=help_msg
    )

    help_msg = "Force overwrite if folder already exists."
    parser.add_argument("--force", action="store_true", help=help_msg)

    args = parser.parse_args(args)

    return parser, args


class GenerateGAPairs():

    def __init__(
        self,
        spectra_path: Path,
        beta: List[float],
        force: bool = False,
        kernel_type: str = "fermionic",
        *args, **kwargs
    ):
        assert spectra_path.is_dir(), "Not a valid directory"
        self.spectra_path = spectra_path

        if "seed" in kwargs:
            self.seed = int(kwargs["seed"])
        else:
            self.seed = 42
        np.random.seed(self.seed)

        with open(spectra_path / Path("config.json"), "r") as f:
            spectra_config = json.load(f)

        self.w_max = spectra_config["w_max"]
        self.w_steps = spectra_config["w_steps"]

        self.n_sym_samples = spectra_config["n_sym_samples"]
        self.n_asym_samples = spectra_config["n_asym_samples"]

        # we currently restrict yourselfs to the sam number
        # of tau and w points.
        self.tau_steps = spectra_config["w_steps"]

        if not isinstance(beta, list):
            beta = [beta]

        self.outpath = self.prepare_outfolder(spectra_path, beta)
        try:
            self.outpath.mkdir(parents=True, exist_ok=force)
        except FileExistsError:
            print("Found GA Pairs: ")
            print(f'"raw_samples_path": "{self.outpath}",')
            raise FileExistsError

        self.kernel_type = kernel_type

        self.beta = self.prepare_beta(beta)

        self.config = spectra_config

        self.save_config(beta)

    def prepare_outfolder(self, spectra_path: Path, beta: List[float]) -> Path:
        outpath = spectra_path.resolve().parents[1]

        # Get proper formatting
        sbeta = [str(float(b)).replace('.', 'c') for b in np.array(beta).round(2)]
        if len(beta) == 1:
            outpath = outpath / f"lambda_{sbeta[0]}"

        elif len(beta) == 2:
            outpath = outpath / f"beta_{sbeta[0]}to{sbeta[1]}_random"

        elif len(beta) == 3:
            outpath = outpath / f"beta_{sbeta[0]}to{sbeta[1]}_{int(beta[2])}steps"

        outpath = outpath / spectra_path.name
        outpath = outpath / f"raw_samples"

        return outpath

    def prepare_beta(self, beta: List[float]) -> np.ndarray:

        tmp_dict = {}
        if len(beta) == 1:
            self.check_beta(beta[0])
            tmp_dict["symmetric"] = np.full(self.n_sym_samples, beta[0])
            tmp_dict["asymmetric"] = np.full(self.n_asym_samples, beta[0])

        elif len(beta) == 2:
            self.check_beta(beta[1])
            rng_list = np.random.rand(self.n_sym_samples) * (beta[1] - beta[0])
            rng_list = beta[0] + rng_list
            tmp_dict["symmetric"] = rng_list.round(2)

            rng_list = np.random.rand(self.n_asym_samples) * (beta[1] - beta[0])
            rng_list = beta[0] + rng_list
            tmp_dict["asymmetric"] = rng_list.round(2)

        elif len(beta) == 3:
            self.check_beta(beta[1])
            values = np.linspace(beta[0], beta[1], int(beta[2])).round(2)
            tmp_dict["symmetric"] = np.random.choice(values, self.n_sym_samples)
            tmp_dict["asymmetric"] = np.random.choice(values, self.n_asym_samples)
        else:
            raise IndexError("Only 1 to 3 arguments are valid for beta")

        self.save_samples(tmp_dict["symmetric"], "symmetric_beta")

        return tmp_dict

    def check_beta(self, beta):

        kernel_num = 2 * np.log(1.7e308) / self.w_max
        kernel_denom = 2 * np.arccosh(1.7e308 / 2.0) / self.w_max

        max_beta = min(kernel_num, kernel_denom)

        if beta > max_beta:
            msg = f"Beta value of {beta} leads to inf in the kernel.\n"
            msg += f"Maximum for w_max={self.w_max} is beta={max_beta}"
            raise ValueError(msg)

    def generate(self):
        '''Creates fermionic samples (G = K*A). Returns a
        list/array? of dictionaries S[n_samples]['A or G'].'''

        self.generate_samples(symmetric=True)
        self.generate_samples(symmetric=False)

        set_permissions(self.outpath)

    def generate_samples(self, symmetric: bool):
        if symmetric:
            filename = "symmetric"
        else:
            filename = "asymmetric"

        spectra = np.load(self.spectra_path / f"{filename}.npy")
        sample, coeffs, coeffs_G = self.calculate_gtau(spectra, self.beta[filename])
        self.save_samples(sample, f"{filename}")
        self.save_samples(coeffs, f"{filename}_coeffs")
        # self.save_samples(coeffs_G, f"{filename}_coeffs_G")
        self.save_samples(self.beta[filename], f"{filename}_beta")

    def calculate_gtau(
        self, spectra: np.ndarray, beta: np.array
    ):
        sample = np.zeros((spectra.shape[0], 2, self.w_steps))
        coeffs = np.zeros((spectra.shape[0], self.w_steps))
        coeffs_G = np.zeros((spectra.shape[0], self.w_steps))

        for ub in np.unique(beta):
            idx = beta == ub

            # if self.config['w_steps']==100:
            #     if self.config['log_grid']==True:
            #         print(f'Using log {self.config["log"]} kernel')
            #         kernel = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
            #         v = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/v.npy')
            #         u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/u.npy')
            #     if self.config['log_grid']==False:
            #         print('Using standard kernel')
            #         kernel = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
            #         v = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100/lambda_{int(beta[0]*self.w_max)}/v.npy')
            #         u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100/lambda_{int(beta[0]*self.w_max)}/u.npy')
            # if self.config['w_steps']==1000:
            #     print('Using standard kernel')
            #     kernel = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
            #     v = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels/lambda_{int(beta[0]*self.w_max)}/v.npy')
            #     u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels/lambda_{int(beta[0]*self.w_max)}/u.npy')

            # if self.config['ctqmc_grid']==True:
            #     if self.config['w_steps']==100:

            # if self.config['log_grid']==False:
            #     print('************* KERNEL OVERRIDE ************* ')
            #     print('Using CTQMC standard kernel')
            #     print('************* KERNEL OVERRIDE ************* ')
            #     if self.config['w_steps']==100:            
            #         kernel = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels_100/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
            #         v = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels_100/lambda_{int(beta[0]*self.w_max)}/v.npy')
            #         u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels_100/lambda_{int(beta[0]*self.w_max)}/u.npy')
            #     if self.config['w_steps']==1000:            
            #         kernel = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
            #         v = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels/lambda_{int(beta[0]*self.w_max)}/v.npy')
            #         u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels/lambda_{int(beta[0]*self.w_max)}/u.npy')

            # if self.config['log_grid']==True:            
            #     print('************* KERNEL OVERRIDE ************* ')
            #     print('Using CTQMC log kernel')
            #     print('************* KERNEL OVERRIDE ************* ')
            #     kernel = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
            #     v = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/v.npy')
            #     u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/ctqmc_kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/u.npy')



            if self.config['log_grid']==False:
                print('************* KERNEL OVERRIDE ************* ')
                print('Using CTQMC standard kernel')
                print('************* KERNEL OVERRIDE ************* ')
                if self.config['w_steps']==100:            
                    kernel = np.load(f'../data_2025_new/ctqmc_kernels_100/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
                    v = np.load(f'../data_2025_new/ctqmc_kernels_100/lambda_{int(beta[0]*self.w_max)}/v.npy')
                    u = np.load(f'../data_2025_new/ctqmc_kernels_100/lambda_{int(beta[0]*self.w_max)}/u.npy')
                if self.config['w_steps']==1000:            
                    kernel = np.load(f'../data_2025_new/ctqmc_kernels_colab/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
                    v = np.load(f'../data_2025_new/ctqmc_kernels_colab/lambda_{int(beta[0]*self.w_max)}/v.npy')
                    u = np.load(f'../data_2025_new/ctqmc_kernels_colab/lambda_{int(beta[0]*self.w_max)}/u.npy')

            if self.config['log_grid']==True:            
                print('************* KERNEL OVERRIDE ************* ')
                print('Using CTQMC log kernel')
                print('************* KERNEL OVERRIDE ************* ')
                kernel = np.load(f'/colab/Projects/work/InvPro/data/ctqmc_kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/kernel.npy')
                v = np.load(f'/colab/Projects/work/InvPro/data/InvPro/ctqmc_kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/v.npy')
                u = np.load(f'/colab/Projects/work/InvPro/data/ctqmc_kernels_100_log{self.config["log"]}/lambda_{int(beta[0]*self.w_max)}/u.npy')
            print('continue')
            # kernel = get_shinaoka_kernel(
            #     self.w_max, self.w_steps, ub, self.tau_steps, self.kernel_type
            # )
            w_grid = np.load(self.spectra_path / f"w_grid.npy")
            Pl = spectra[idx, None,:] * kernel
            Pl = np.einsum('abc->acb', Pl)
            G = np.trapz(y=Pl, x=w_grid, axis=1)

            # G = np.einsum('ca,ba->cb', spectra[idx], kernel)
            # # Grid spacing: 2 * _omega_max / (w_steps-1) )
            # G = G * (2 * self.w_max / (self.w_steps - 1))

            sample[idx, 0] = G
            sample[idx, 1] = spectra[idx]
            coeffs[idx] = self.calculate_coefficients(spectra[idx], kernel, v)
            coeffs_G[idx] = self.calculate_coefficients_G(G[idx], kernel, u)
    
        return sample, coeffs, coeffs_G

    def calculate_coefficients(self, spectra: np.ndarray, kernel: np.ndarray, v: np.array):

        # v = get_svd(kernel)
        # np.save(self.outpath / f"vectors", v)
        # np.save(self.outpath / f"kernel", kernel)
        coeffs = np.einsum('ij,kj->ki', v, spectra)

        return coeffs

    def calculate_coefficients_G(self, G: np.ndarray, kernel: np.ndarray, u: np.array):

        # u = get_svd_u(kernel)
        # np.save(self.outpath / f"u_vectors", u)
        coeffs = np.einsum('ji,kj->ki', u, G)

        return coeffs

    def save_samples(self, samples: np.ndarray, outname: str):
        # ::: Save spectra into file
        np.save(self.outpath / f"{outname}", samples)

    def save_config(self, beta: str):
        config = {
            "spectra": str(self.spectra_path.resolve()),
            "kernel_type": self.kernel_type,
            "beta": beta,
            "tau_steps": self.tau_steps,
            "w_max": self.w_max,
            "w_steps": self.w_steps,
            "seed": self.seed
        }
        with open(self.outpath / f"config.json", "w") as f:
            json.dump(config, f, indent=4)


def main(args):
    parser, args = create_parser(args)
    GP = GenerateGAPairs(
        spectra_path=args.spectra_path,
        beta=args.beta,
        force=args.force,
        log=args.log,
        log_grid=args.log_grid
    )
    GP.generate()


if __name__ == '__main__':
    main(sys.argv[1:])
