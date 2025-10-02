import argparse
import logging
from pathlib import Path

from xlstm_scaling_laws.analysis.parametric_sclaw_fit.run_fit_grid import (
    combine_fit_grid_results_into_df,
    filename_from_params,
    load_scaling_law_fit_grid_result,
    params_from_filename,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit_grids_dir",
        type=str,
        required=True,
        help="The directory where the full fit grid results are stored. Loads the results from this directory.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=False,
        help="The number of top k fits to include in the combined dataframe.",
        default=100,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="The directory where the combined dataframe will be saved.",
        default=None,
    )

    args = parser.parse_args()
    LOGGER.info(args)


    fit_grids_dir = args.fit_grids_dir
    topk = args.topk
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / "data_lnd_fits"

    LOGGER.info(f"Output directory: {output_dir}")

    fit_grids_dir = Path(fit_grids_dir)

    for fit_grid_file in fit_grids_dir.glob("*.pkl"):

        LOGGER.info(f"Processing file: {fit_grid_file.name}")
        # Extract the parameters from the filename
        params = params_from_filename(fit_grid_file.name)
        LOGGER.info(f"Extracted parameters from filename: {params}")

        # Load the fit grid results
        fit_grid_results = load_scaling_law_fit_grid_result(save_dir=fit_grids_dir, **params)

        # Combine the fit grid results into a DataFrame
        combined_fit_grid_df = combine_fit_grid_results_into_df(
            grid_result=fit_grid_results,
            topk=topk,
            sort_by_col=("optim_results", "loss"), # Note: we sort by the objective / loss function
            ascending=True,
        )

        filename = f"combined_df__{filename_from_params(**params)}"

        # Save the combined fit grid results to a pickle file
        combined_fit_grid_df.to_pickle(output_dir / filename)
        LOGGER.info(f"Saved combined fit grid results to: {output_dir / filename}")



