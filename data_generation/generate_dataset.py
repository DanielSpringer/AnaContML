import argparse
import json
import numpy as np
from pathlib import Path
import sys
from utils import set_permissions
import timeit
from copy import deepcopy
from kernels import get_svd_u


def create_parser(args):
    parser = argparse.ArgumentParser()

    help_msg = "Path to the G(tau)/A(w) pairs generated for a specific "
    help_msg += "beta/w_max setting. E.g Path('w_max-5/beta-200/test'). "
    help_msg += "Note that `/test` is a folder that is generated based "
    help_msg += "on the given `description` in the first step."
    parser.add_argument(
        "--raw_samples_path", type=Path, default="", help=help_msg
    )

    help_msg = "Gaussian noise amplitude to be applied pointwise "
    help_msg += "for each Green's function."
    parser.add_argument("--noise_level", type=float, default=None, help=help_msg)

    help_msg = "Number of noisy G(tau) functions sampled with the given noise level."
    parser.add_argument("--noisy_samples", type=int, default=None, help=help_msg)

    help_msg = "Force overwrite if folder already exists."
    parser.add_argument("--force", action="store_true", help=help_msg)

    args = parser.parse_args(args)

    return parser, args


class GenerateDataset():

    def __init__(
        self,
        raw_samples_path: Path,
        noise_level: float,
        noisy_samples: int,
        force: bool = False,
        noise_type = 'standard',
        *args,
        **kwargs
    ):

        assert raw_samples_path.is_dir(), f"directory {raw_samples_path} not found"
        self.raw_samples_path = raw_samples_path

        self.noise_level = noise_level
        self.noisy_samples = noisy_samples

        self.outpath = self.prepare_outfolder(raw_samples_path)
        self.outpath.mkdir(parents=True, exist_ok=force)

        self.noise_type = noise_type

        if "seed" in kwargs:
            self.seed = int(kwargs["seed"])
        else:
            self.seed = 42
        np.random.seed(self.seed)

        self.save_config()

    def prepare_outfolder(self, raw_samples_path: Path) -> Path:

        noise_level = str(self.noise_level).replace('.', 'c')
        noisy_samples = str(self.noisy_samples).replace('.', 'c')

        tmp = f"noise_level_{noise_level}_noisy_samples_{noisy_samples}"
        folder = raw_samples_path.parents[0] / tmp

        return folder

    def generate_samples(self, symmetric: bool):

        if symmetric:
            filename = "symmetric"
        else:
            filename = "asymmetric"

        starttime = timeit.default_timer()

        raw_samples = np.load(self.raw_samples_path / f"{filename}.npy")

        noisy_samples = []
        noisy_samples_coeffs = []
        noisy_samples_F_coeffs = []
        for i in range(self.noisy_samples):
            noisy_samples.append(self.add_noise(raw_samples))
            noisy_samples_coeffs.append(self.calculate_coefficients_G(noisy_samples))
            noisy_samples_F_coeffs.append(self.calculate_F_coefficients_G(noisy_samples))
        noisy_samples = np.concatenate(np.array(noisy_samples))

        self.save_samples(noisy_samples, f"{filename}")
        self.save_betas(filename)
        self.save_coeffs(filename)

        print("Sample generation:", timeit.default_timer() - starttime)

    def generate_samples_positive(self, symmetric: bool):

        if symmetric:
            filename = "symmetric"
        else:
            filename = "asymmetric"

        starttime = timeit.default_timer()

        raw_samples = np.load(self.raw_samples_path / f"{filename}.npy")

        noisy_samples = []
        # noisy_samples_coeffs = []
        # noisy_samples_F_coeffs = []
        for i in range(self.noisy_samples):
            # print(self.add_noise(raw_samples).shape)


            if self.noise_type == "standard":
                print('Using standard noise')
                c_sample = self.add_noise(raw_samples)
            if self.noise_type == "ctqmc_1":
                print('Using ctqmc 1 noise')
                c_sample = self.add_noise_CTQMC_1(raw_samples)
            if self.noise_type == "ctqmc_2":
                print('Using ctqmc 2 noise')
                c_sample = self.add_noise_CTQMC_2(raw_samples)
            if self.noise_type == "ctqmc_3":
                print('Using ctqmc 3 noise')
                c_sample = self.add_noise_CTQMC_3(raw_samples)
            if self.noise_type == "ctqmc_4":
                print('Using ctqmc 4 noise')
                c_sample = self.add_noise_CTQMC_4(raw_samples)

            c_sample[c_sample<0] = 0
            noisy_samples.append(c_sample)
            # noisy_samples_coeffs.append(self.calculate_coefficients_G(c_sample))
            # noisy_samples_F_coeffs.append(self.calculate_F_coefficients_G(c_sample))
        noisy_samples = np.concatenate(np.array(noisy_samples))
        # noisy_samples_coeffs = np.concatenate(np.array(noisy_samples_coeffs))
        # noisy_samples_F_coeffs = np.concatenate(np.array(noisy_samples_F_coeffs))

        self.save_samples(noisy_samples, f"{filename}")
        # self.save_samples(noisy_samples_coeffs, f"{filename}_coeffs_G")
        # self.save_samples(noisy_samples_F_coeffs, f"{filename}_F_coeffs_G")
        self.save_betas(filename)
        self.save_coeffs(filename)

        print("Sample generation:", timeit.default_timer() - starttime)

    # def calculate_coefficients_G(self, G: np.ndarray):

    #     u = np.load(f'/iarai/home/daniel.springer/Projects/InvPro/kernels_100_log/lambda_{0}/u.npy')
    #     # u = np.load(self.raw_samples_path / "u_vectors.npy")
    #     coeffs = np.einsum('ji,kj->ki', u, G[:,0,:])
    #     return coeffs

    # def calculate_F_coefficients_G(self, G: np.ndarray):

    #     coeffs = np.fft.fft(G[:,0,:])
    #     return coeffs

    def save_betas(self, filename: str):
        betas = [np.load(self.raw_samples_path / f"{filename}_beta.npy")]
        betas = np.concatenate(np.array(betas * self.noisy_samples))
        self.save_samples(betas, f"{filename}_beta")

    def save_coeffs(self, filename: str):
        coeffs = [np.load(self.raw_samples_path / f"{filename}_coeffs.npy")]
        coeffs = np.concatenate(np.array(coeffs * self.noisy_samples))
        self.save_samples(coeffs, f"{filename}_coeffs")

    def add_noise(self, raw_samples: np.ndarray) -> np.ndarray:
        noise = np.random.normal(
            0, self.noise_level, (raw_samples.shape[0], raw_samples.shape[2])
        )

        noisy_samples = deepcopy(raw_samples)
        noisy_samples[:, 0, :] += noise

        return noisy_samples

    def add_noise_CTQMC_1(self, raw_samples: np.ndarray) -> np.ndarray:
        noisy_samples = deepcopy(raw_samples)
        x = np.linspace(0,raw_samples.shape[2],raw_samples.shape[2])

        for n in range(raw_samples.shape[0]):
            noise = np.random.normal(0, self.noise_level, raw_samples.shape[2])
            noise = noise * np.minimum(raw_samples[n,0]*1000,1)
            noisy_samples[n, 0, :] += noise
            raw_samples
            tau1 = 0
            tau2 = 1
            k = 0
            while k==0 and tau1<raw_samples.shape[2]:
                if raw_samples[n,0,tau1]<0.05:
                    k=1
                tau1+=1
            k = 0
            while k==0 and tau2<raw_samples.shape[2]:
                if (np.abs(raw_samples[n,0,tau2-1]-raw_samples[n,0,tau2]))<1e-5:
                    k=1
                tau2+=1
            if tau2>tau1:
                sigma = (tau2-tau1)/3
                sl = np.exp(-((tau2-tau1)-x)**2/(2*sigma**2)) / 2
                sr = np.exp(-(raw_samples.shape[2]-(tau2-tau1)-x)**2/(2*sigma**2)) / 2
            for tau in range(raw_samples.shape[2]):
                draw = np.random.rand()
                if (draw<sl[tau] or draw<sr[tau]):
                    spike = np.minimum(0.01, np.random.rand()*1e-2*(1/noisy_samples[n, 0, tau])/10000)
                    noisy_samples[n, 0, tau] = noisy_samples[n, 0, tau] + spike
        return noisy_samples

    def add_noise_CTQMC_2(self, raw_samples: np.ndarray) -> np.ndarray:
        noisy_samples = deepcopy(raw_samples)
        x = np.linspace(0,raw_samples.shape[2],raw_samples.shape[2])

        for n in range(raw_samples.shape[0]):
            noise = np.random.normal(0, self.noise_level, raw_samples.shape[2])
            noise = noise * np.minimum(raw_samples[n,0]*1000,1)
            noisy_samples[n, 0, :] += noise
            raw_samples
            tau1 = 0
            tau2 = 1
            k = 0
            while k==0 and tau1<raw_samples.shape[2]:
                if raw_samples[n,0,tau1]<0.05:
                    k=1
                tau1+=1
            k = 0
            while k==0 and tau2<raw_samples.shape[2]:
                if (np.abs(raw_samples[n,0,tau2-1]-raw_samples[n,0,tau2]))<1e-5:
                    k=1
                tau2+=1
            if tau2>tau1:
                sigma = (tau2-tau1)/2
                sl = np.exp(-((tau2-tau1)-x)**2/(2*sigma**2)) / 2
                sr = np.exp(-(raw_samples.shape[2]-(tau2-tau1)-x)**2/(2*sigma**2)) / 2
                sl = sl / np.trapz(x=x, y=sl)
                sr = sr / np.trapz(x=x, y=sl)
                for tau in range(raw_samples.shape[2]):
                    draw = np.random.rand()
                    if (draw<sl[tau] or draw<sr[tau]):
                        spike = np.minimum(0.01, np.random.rand()*1e-2*(1/noisy_samples[n, 0, tau])/10000)
                        noisy_samples[n, 0, tau] = noisy_samples[n, 0, tau] + spike
        return noisy_samples


    def add_noise_CTQMC_3(self, raw_samples: np.ndarray) -> np.ndarray:
        noisy_samples = deepcopy(raw_samples)

        for n in range(raw_samples.shape[0]):
            # noise = noise * np.minimum(raw_samples[n,0]*1000,1)
            # print((noisy_samples[n, 0, :]>self.noise_level).sum())
            noise = np.random.normal(0, self.noise_level, (noisy_samples[n, 0, :]>self.noise_level).sum())
            noisy_samples[n, 0, noisy_samples[n, 0, :]>self.noise_level] += noise

            noise = np.random.normal(0, self.noise_level/10, (noisy_samples[n, 0, :]>self.noise_level/10).sum())
            noisy_samples[n, 0, noisy_samples[n, 0, :]>self.noise_level/10] += noise

            noise = np.random.normal(0, self.noise_level/100, (noisy_samples[n, 0, :]>self.noise_level/100).sum())
            noisy_samples[n, 0, noisy_samples[n, 0, :]>self.noise_level/100] += noise
        return noisy_samples


    def add_noise_CTQMC_4(self, raw_samples: np.ndarray) -> np.ndarray:
        noisy_samples = deepcopy(raw_samples)

        for n in range(raw_samples.shape[0]):
            # noise = noise * np.minimum(raw_samples[n,0]*1000,1)
            # print((noisy_samples[n, 0, :]>self.noise_level).sum())
            noise = np.random.normal(0, self.noise_level, (noisy_samples[n, 0, :]>self.noise_level).sum())
            noisy_samples[n, 0, noisy_samples[n, 0, :]>self.noise_level] += noise

            noise = np.random.normal(0, self.noise_level/10, (noisy_samples[n, 0, :]>self.noise_level/10).sum())
            noisy_samples[n, 0, noisy_samples[n, 0, :]>self.noise_level/10] += noise

            noise = np.random.normal(0, self.noise_level/100, (noisy_samples[n, 0, :]>self.noise_level/100).sum())
            noisy_samples[n, 0, noisy_samples[n, 0, :]>self.noise_level/100] += noise
        return noisy_samples

    def generate(self):

        # self.generate_samples(symmetric=True)
        # self.generate_samples(symmetric=False)
        self.generate_samples_positive(symmetric=True)
        self.generate_samples_positive(symmetric=False)

        set_permissions(self.outpath)

    def save_samples(self, samples: np.ndarray, outname: str):
        # ::: Save spectra into file
        np.save(self.outpath / f"{outname}", samples)

    def save_config(self):
        config = {
            "noise_level": self.noise_level,
            "noisy_samples": self.noisy_samples,
            "seed": self.seed
        }
        with open(self.outpath / f"config.json", "w") as f:
            json.dump(config, f, indent=4)


def main(args):
    parser, args = create_parser(args)

    GD = GenerateDataset(
        raw_samples_path=args.raw_samples_path,
        noise_level=args.noise_level,
        noisy_samples=args.noisy_samples,
        force=args.force
    )
    GD.generate()


if __name__ == '__main__':
    main(sys.argv[1:])
