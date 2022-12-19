# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare ProDoc and CirkaFlow

# %%
import pandas as pd

import njab
import src

import config

# %%
FOLDER = config.folder_reports
FOLDER

# %%
files_in = {
    'olink': config.fname_pkl_all_olink,
    'clinic': config.fname_pkl_all_clinic_num
}
files_out = dict()

# %%
clinic = pd.read_pickle(files_in['clinic'])


# %%
TEST_IDS = clinic.filter(like='Cflow', axis=0).index.to_list()

clinic_cflow = clinic.loc[TEST_IDS]
clinic_prodoc = clinic.drop(TEST_IDS)

clinic_prodoc.shape, clinic_cflow.shape

# %%
col = 'MELD-score'

bins = range(int(clinic[col].min()), int(clinic[col].max() + 1), 3)

ax = clinic_prodoc[col].rename('ProDoc').hist(alpha=0.9, legend=True, bins=bins)
ax = clinic_cflow[col].rename('CirkaFlow').hist(alpha=0.7,
                                                legend=True,
                                                bins=bins)
_ = ax.set_xlabel(col)
_ = ax.set_ylabel('n observations in bin')
_ = ax.set_xticks(list(bins))

fname = FOLDER / 'hist_meld_score_cohorts.pdf'
files_out[fname.name] = fname
njab.plotting.savefig(fig=ax.get_figure(), name=fname)

# %%
ax = clinic_prodoc[col].rename('ProDoc').hist(alpha=0.9,
                                              legend=True,
                                              density=1,
                                              bins=bins)
ax = clinic_cflow[col].rename('CirkaFlow').hist(alpha=0.7,
                                                legend=True,
                                                density=1,
                                                bins=bins)
_ = ax.set_xlabel(col)
_ = ax.set_ylabel('n observations in bin')
_ = ax.set_xticks(list(bins))

fname = FOLDER / 'hist_meld_score_cohorts_normalized.pdf'
files_out[fname.name] = fname
njab.plotting.savefig(fig=ax.get_figure(), name=fname)

# %%
src.io.print_files(files_out)
