import argparse
import json
from generate_spectra import GenerateSpectra
from generate_ga_pairs import GenerateGAPairs
from generate_dataset import GenerateDataset
from pathlib import Path
import sys


def create_parser(args):
    parser = argparse.ArgumentParser()

    help_msg = "Path to which the generated samples should be written to."
    parser.add_argument(
        "--outpath", type=Path, default="", help=help_msg
    )

    help_msg = "Path to configuration file for the sample generation."
    parser.add_argument(
        "--config", type=Path, default="", help=help_msg
    )

    help_msg = "Path to spectra to start producing samples from."
    parser.add_argument(
        "--spectra", type=Path, default="", help=help_msg
    )

    help_msg = "Path to raw GA pairs to start producing noisy samples from."
    parser.add_argument(
        "--raw_samples", type=Path, default="", help=help_msg
    )

    help_msg = "Force overwrite if folder already exists."
    parser.add_argument("--force", action="store_true", help=help_msg)

    args = parser.parse_args(args)

    return parser, args


def main(args):
    parser, args = create_parser(args)

    assert args.outpath.is_dir()
    assert args.config.is_file()

    with open(args.config, "r") as f:
        config = json.load(f)

    config["outpath"] = args.outpath
    config["force"] = args.force

    from_spectra = False
    from_raw_samples = False
    if "spectra" in str(args.spectra):
        config["spectra_path"] = args.spectra
        from_spectra = True

    if "raw_samples" in str(args.raw_samples):
        config["raw_samples_path"] = args.raw_samples
        from_spectra = False
        from_raw_samples = True

    if not from_spectra and not from_raw_samples:
        try:
            print("Generating Spectra")
            GS = GenerateSpectra(**config)
            config["spectra_path"] = GS.outpath
            GS.generate()
            del GS
        except FileExistsError:
            msg = "Please specify `spectra_path` in the config if "
            msg += "you do not want to use --force option or use "
            msg += "'generate_spectra.py'"
            print(msg)
            exit()
    elif from_spectra:
        print("Found 'spectra_path'. Skipping spectra generation.")
    elif from_raw_samples:
        print("Found 'raw_samples_path'. Skipping spectra generation.")
    else:
        print("Dont know what to do...")
        exit()

    if not from_raw_samples:
        try:
            print("Generating GA Pairs")
            GP = GenerateGAPairs(**config)
            config["raw_samples_path"] = GP.outpath
            GP.generate()
            del GP
        except FileExistsError:
            msg = "Please specify `raw_samples_path` in the config if "
            msg += "you do not want to use --force option or use "
            msg += "'generate_ga_pairs.py'"
            print(msg)
            exit()
    elif from_raw_samples:
        print("Found 'raw_samples_path'. Skipping GA generation.")
    else:
        print("Dont know what to do...")
        exit()

    try:
        print("Generating Dataset")
        GD = GenerateDataset(**config)
        GD.generate()
        del GD
    except FileExistsError:
        msg = "Dataset already exists. "
        msg += "Use 'generate_dataset.py' and not 'run_all.py' "
        msg += "if you want to rewrite it..."
        print(msg)
        exit()


if __name__ == '__main__':
    main(sys.argv[1:])
