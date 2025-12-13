# TL;DR

Produce noisy G/A pairs based on the information in `config.json`.
```bash
python run_all.py --outpath ./ --config config.json --force
``` 
The numpy random seed is `42`. 

# How to use

Three scripts are provided to generate synthetic data. 

- `generate_spectra.py` Generates a set of spectra based mainly on the following input parameters:
    - `w_max: float` The grid of the spectrum will range from `-w_max` to `w_max`
    - `w_steps_ int` The grid will have `w_steps` between  `-w_max` and `w_max`
    - `sigma: List[float]`: Specifies the minimum and maximum value of the width for each gaussian peak.
    - `n_peaks: int`: Specifies the maximum number of peaks within the spectrum.
    - `n_sym_samples: int`: Number of symmetric spectra to produce
    - `n_asym_samples: int`: Number of symmetric spectra to produce
    - `boundary_eps: float`: If the values at the boundary are above the threshold a new sample is drawn.
    - `sampling_policy: str`: Definition of the spectra generation process. Spectra are written in a folder with this name. The default process is `springer_static_spectrum`. To implement a new process define a member function named `_{sampling_policy}_spectrum(...)` which takes `self` and `symmetric` as argument.
    - `force: bool`: Overwrite samples if they exist.

Example:
```bash
python generate_spectra.py --outpath ./ --w_max 5 --w_steps 1000 --sigma 0.1 0.9 --n_peaks 4 --n_sym_samples 10 --n_asym_samples 10 --boundary_eps 1e-5 --sampling_policy springer_static
```
---
- `generate_ga_pairs.py` Generates the corresponding Green's function to each spectrum generated in the previous step based on the following parameters. Additionally, the coefficients of the individual spectra are calculated and written in a separate file and the beta values are also stored separately.
    - `spectra_path: Path`: Path to the directory generated for one specific `w_max` in the previous step. E.g `w_max_5c0_w_steps_1000/n_peaks_4_sigma_0c1_0c9/spectra/10_sym_10_asym_springer_static`. Note that the path must be given as a Path object
    - `beta: List[float]`: Beta value which is used in combination with `w_max` to calculate the corresponding Kernel. Note that `tau_steps` are chosen to be the same as `w_steps`. This value can be given in three different ways:
        - One value, e.g. `--beta BETA`: All Green's functions are calculated by using the same `beta` value in the kernel.
        - Two values e.g. `--beta BETA_MIN BETA_MAX`: The values for `beta` are sampled randomly within the range `BETA_MIN - BETA_MAX` and the Green's function is calculated using the sampled `beta` value in the kernel.
        - Three values e.g. `--beta BETA_MIN BETA_MAX STEPS`: The range `BETA_MIN - BETA_MAX` is dividied into evenly spaced numbers based on `STEPS`. The Green's function is calculated by randomly sampling one of the resulting `beta` values within this range. 
    - `force: bool`: Overwrite samples if they exist.

Example:
```bash
python generate_ga_pairs.py --spectra_path w_max_5c0_w_steps_1000/n_peaks_4_sigma_0c1_0c9/spectra/10_sym_10_asym_springer_static --beta 200
```
---
- `generate_dataset.py` Generates Green's functions modified with Gaussian noise based on the samples generated in the previous step based on the following parameters:
    - `raw_samples_path: Path`: Path to the G(tau)/A(w) pairs generated for a specific `beta`/`w_max` setting. E.g `w_max_5c0_w_steps_1000/n_peaks_4_sigma_0c1_0c9/beta_200c0/springer/raw_samples`.
    - `noise_level: float`: Gaussian noise amplitude to be applied pointwise for each Green's function.
    - `noisy_samples: int`: Number of noisy Green's function to be sampled from the smooth Green's function given as input.
    - `force: bool`: Overwrite samples if they exist.

Example:
```bash
python generate_dataset.py --raw_samples_path w_max_5c0_w_steps_1000/n_peaks_4_sigma_0c1_0c9/beta_200c0/10_sym_10_asym_springer_static/raw_samples --noise_level 0.01 --noisy_samples 10
```
---

The resulting folder structure looks as follows. Values given as `{value}` are based on the parameters given above:
```
w_max_{w_max_1}_w_steps_{w_steps}
├── n_peaks_{n_peaks}_sigma_{sigma}
|   ├── spectra
|   |   └── {n_sym_samples}_sym_{n_asym_samples}_asym_{sampling_policy}
|   |           symmetric.npy
|   |           asymmetric.npy
|   |           config.json
|   |     
|   ├── beta_{beta_1}
|   |   └── {n_sym_samples}_sym_{n_asym_samples}_asym_{sampling_policy}
|   |       ├── raw_samples
|   |       |       symmetric.npy
|   |       |       symmetric_beta.npy
|   |       |       symmetric_coeffs.npy
|   |       |       asymmetric.npy
|   |       |       asymmetric_beta.npy
|   |       |       asymmetric_coeffs.npy
|   |       |       config.json
|   |       |
|   |       ├── noise_level_{eta_1}_noisy_samples_{noisy_samples}
|   |       |       symmetric.npy
|   |       |       symmetric_beta.npy
|   |       |       symmetric_coeffs.npy
|   |       |       asymmetric.npy
|   |       |       asymmetric_beta.npy
|   |       |       asymmetric_coeffs.npy
|   |       |       config.json
|   |       |
|   |       └── noise_level_{eta_2}_noisy_samples_{noisy_samples}
|   |           └── ...
|   |
|   └── beta_{beta_2}
|       └── ...
|
w_max_{w_max_2}
    └── n_peaks_{n_peaks}_sigma_{sigma}
        └──  ...


```

# Spectrum policies

Definition of the implemented policies used for spectrum generation.

## Springer Static (`--sampling_policy springer_static`)

- The position of each gaussian peak is sampled within a region of `w_max - 2 * max_width` where `max_width == sigma[1]`. This should guarantee that the full `w_max` range is utilized and vanishes at the border.
- The width of each gaussian peak is sampled between the given minimum and maximum width (`sigma[0]` and `sigma[1]`). 
- For symmetric spectra, the number of peaks is `2*n_peaks` since the spectrum is mirrored around the origin.

## Springer Broadening (`--sampling_policy springer_broadening`)

- The position of each gaussian peak is sampled within a region of `w_max - 2 * max_width` where `max_width == sigma[1]`. This should guarantee that the full `w_max` range is utilized and that the spectrum vanishes at the border.
- The width of each gaussian peak is sampled between the given minimum and maximum width (`sigma[0]` and `sigma[1]`) and scaled based on the position of the peak. The scaling (`width * (1 + 2 * (abs(pos) / w_max))`) is chosen such that peaks farther away from the origin are broader.
- For symmetric spectra, the number of peaks is `2*n_peaks` since the spectrum is mirrored around the origin.

## Fournier (`--sampling_policy fournier`)

- The first peak is always sampled within a region of `-0.5` and `0.5`.
- The position of the other gaussian peaks is sampled within a region of `-position` and `position` where `position=(6.0 * w_max) / 15.0`. This scaling factor is chosen such that it reproduces the sampling from the Fournbier paper.
- The width of each gaussian peak is sampled between the given minimum and maximum width (`sigma[0]` and `sigma[1]`). 
- For symmetric spectra, the number of peaks is `2*n_peaks` since the spectrum is mirrored around the origin.