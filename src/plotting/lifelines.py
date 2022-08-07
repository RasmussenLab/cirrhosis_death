"""
Lifeline plots. Adapted from 
https://allendowney.github.io/SurvivalAnalysisPython/02_kaplan_meier.html
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_lifelines(obs:pd.DataFrame, ax=None, start_col='DateDiagnose', end_col='DateDeath', status_col='dead',):
    """Plot a line for each observation.
    
    obs: DataFrame
    """
    if ax is None:
        fig, ax = plt.subplots()
    for i, (label, row) in enumerate(obs.iterrows()):
        start = row[start_col]
        end = row[end_col]
        status = row[status_col]

        if not status:
            # ongoing
            ax.hlines(i, start, end, color='C2')
        else:
            # complete
            ax.hlines(i, start, end, color='C1')
            ax.plot(end, i, marker='o', color='C1')
    return