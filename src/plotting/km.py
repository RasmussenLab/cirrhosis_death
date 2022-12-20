from matplotlib.axes import Axes
from lifelines import KaplanMeierFitter
import pandas as pd


def compare_km_curves(
    time: pd.Series,
    y: pd.Series,
    pred: pd.Series,
    ax: Axes = None,
    ylim: tuple[int] = (0, 1),
    xlim: tuple[int] = (0, 180),
    xlabel: str = None,
    ylabel: str = None,
) -> Axes:
    """Compare Kaplan-Meier curves for two groups (e.g. based on binary prediction)

    Parameters
    ----------
    time : pd.Series
        Time to event variable
    y : pd.Series
        event variable
    pred : pd.Series
        mask for two groups, e.g. predictions
    ax : Axes, optional
        matplotlib Axes object, by default None
    ylim : tuple[int], optional
        y-axis bounds, by default (0, 1)
    xlim : tuple[int], optional
        time-axis bounds, by default (0, 730)
    xlabel : str, optional
        time-axis label, by default None
    ylabel : str, optional
        y-axis label, by default None


    Returns
    -------
    Axes
        _description_
    """
    kmf = KaplanMeierFitter()
    pred = pred.astype(bool)

    mask = ~pred
    kmf.fit(time.loc[mask], event_observed=y.loc[mask])
    ax = kmf.plot(xlim=xlim, ylim=ylim, legend=False, ax=ax)

    mask = pred
    kmf.fit(time.loc[mask], event_observed=y.loc[mask])
    ax = kmf.plot(xlim=xlim,
                  ylim=ylim,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  legend=False)
    return ax