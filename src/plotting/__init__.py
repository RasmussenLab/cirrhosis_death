import matplotlib.pyplot as plt
import seaborn

import njab

from src.plotting.lifelines import plot_lifelines
from src.plotting.km import compare_km_curves

__all__ = ['plot_lifelines', 'compare_km_curves']

plt.rcParams['figure.figsize'] = [4.0, 2.0]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['figure.dpi'] = 147

njab.plotting.set_font_sizes('x-small')

seaborn.set_style("whitegrid")
