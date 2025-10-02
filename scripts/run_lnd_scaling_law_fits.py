import argparse

from xlstm_scaling_laws.analysis.parametric_sclaw_fit.run_fit_grid import run_fit_grids
from xlstm_scaling_laws.utils import setup_output_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param_combination_mode",
        type=str,
        required=True,
        help="The parameter combination mode to use. Options are: 'all', 'single'. If 'single' the parameters are specified in the arguemnts.",
    )
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of scaling law fits.",
    )

    args = parser.parse_args()
    print(args)

    param_combination_mode = args.param_combination_mode

    output_folder = setup_output_folder(
        output_dir="./outputs_lnd_fits",
        name_suffix=args.folder_suffix,
    )
    print(f"Output folder: {output_folder}")

    run_fit_grids(save_dir=output_folder, param_combination_mode=param_combination_mode)
    print(f"Finished running fit grid with mode {param_combination_mode}.")
