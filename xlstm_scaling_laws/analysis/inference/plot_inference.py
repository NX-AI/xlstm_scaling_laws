import pickle

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def load_df(path) -> pd.DataFrame:
    data = pickle.load(open(path, "rb"))
    df = pd.DataFrame(data)
    return df


def get_inference_plot(df: pd.DataFrame, fig: Figure = None, ax: Axes = None) -> None:

    if fig is None and ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    return fig
