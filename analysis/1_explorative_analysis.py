# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Explorative Analysis

# %%
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import pingouin as pg
import seaborn
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve

import src
from src.sklearn import run_pca, StandardScaler
from src.sklearn.scoring import ConfusionMatrix

import config
import njab

# %% [markdown]
# # Set parameters

# %% tags=["parameters"]
TARGET = 'liverDead090infl'
FOLDER = ''
CLINIC=config.fname_pkl_clinic
OLINK=config.fname_pkl_olink
val_ids:str='' # List of comma separated values or filepath
#
clinic_cont=config.clinic_data.vars_cont # list or string of csv, eg. "var1,var2"
clinic_binary=config.clinic_data.vars_binary # list or string of csv, eg. "var1,var2" 
da_covar='Sex,Age,Cancer,Depression,Psychiatric,Diabetes,HeartDiseaseTotal,Hypertension,HighCholesterol' # List of comma separated values or filepath

# %%
# # compare ProDoc train and validation split
# TARGET = 'is_valdiation_sample'
# CLINIC=config.fname_pkl_prodoc_clinic
# OLINK=config.fname_pkl_prodoc_olink
# clinic_cont=('Age,IgM,IgG,IgA,Hgb,Leucocytes,Platelets,Bilirubin,Albumin,CRP,pp,INR,ALAT,MELD-score,MELD-Na,ChildPugh,')
# clinic_binary=('Sex,EtiAlco,EtiFat,EtiHBV,EtiHCV,EtiPBC,EtiAIH,EtiMTX,EtiOther,EtiUnknown,DecomensatedAtDiagnosis,Ascites,EsoBleeding,HRS,HE,Icterus,SBP,'
#                'Hypertension,HighCholesterol,Cancer,Depression,Psychiatric,Diabetes,InsulinDependent,Statins,NonselectBetaBlock'
#                'dead090infl,dead180infl,liverDead090infl,liverDead180infl,hasAdm180,hasAdm90,hasLiverAdm90,hasLiverAdm180')

# %%
if not FOLDER:
    FOLDER = Path(config.folder_reports) / TARGET
    FOLDER.mkdir(exist_ok=True, parents=True)
FOLDER

# %%
clinic = pd.read_pickle(CLINIC)
cols_clinic = src.pandas.get_colums_accessor(clinic)
olink = pd.read_pickle(OLINK)

# %%
pd.crosstab(clinic.DiagnosisPlace, clinic[TARGET], margins=True)


# %%
def join(l): return ','.join([str(x) for x in l])

from typing import Union
def check_isin_clinic(l:Union[list, str]):
    """Remove item from passed list and warn."""
    if isinstance(l, str):
        l = l.split(',')
    ret = list()
    for _var in l:
        if _var not in clinic.columns:
            logging.warning(f"Desired variable not found: {_var}", stacklevel=0)
            continue
        ret.append(_var)
    return ret

# da_covar='Sex,Age,Cancer,missingvariable' # List of comma separated values or filepath
covar = check_isin_clinic(da_covar)
covar

# %%
config.clinic_data.vars_cont = check_isin_clinic(config.clinic_data.vars_cont)

# %%
config.clinic_data.vars_binary = check_isin_clinic(config.clinic_data.vars_binary)

# %% [markdown]
# # Differences between groups defined by target

# %%
clinic

# %%
target_counts = clinic[TARGET].value_counts()

if target_counts.sum() < len(clinic):
    print(
        f"Target has missing values. Can only use {target_counts.sum()} of {len(clinic)} samples."
    )
    mask = clinic[TARGET].notna()
    clinic, olink = clinic.loc[mask], olink.loc[mask]

target_counts

# %%
pd.crosstab(clinic[TARGET], clinic["DecomensatedAtDiagnosis"])

# %%
happend = clinic[TARGET].astype(bool)

# %% [markdown]
# ## Continous

# %%
var = 'Age'
# import scipy.stats
# scipy.stats.ttest_ind(clinic.loc[happend, var], clinic.loc[~happend, var], equal_var=False) # same results as pengoin
pg.ttest(clinic.loc[happend, var], clinic.loc[~happend, var])

# %%
vars_cont = config.clinic_data.vars_cont
ana_differential = njab.stats.groups_comparision.diff_analysis(
    clinic[vars_cont],
    happend,
    event_names=(TARGET, 'no event'),
)
ana_differential = ana_differential.sort_values(('ttest', 'p-val'))

writer = pd.ExcelWriter(FOLDER / '1_differential_analysis.xlsx')
ana_differential.to_excel(writer, "clinic continous")

ana_differential

# %% [markdown]
# ## Binary

# %%
clinic[config.clinic_data.vars_binary].describe()

# %% [markdown]
# Might focus on discriminative power of
#   - DecompensatedAtDiagnosis
#   - alcohol consumption
#
# but the more accute diseases as heart disease and cancer seem to be distinctive

# %%
diff_binomial = []
for var in config.clinic_data.vars_binary[1:]:
    diff_binomial.append(
        njab.stats.groups_comparision.binomtest(clinic[var],
                            happend,
                            event_names=(TARGET, 'no-event')))
for var in config.clinic_data.vars_binary_created:
    diff_binomial.append(
        njab.stats.groups_comparision.binomtest(clinic[var],
                            happend,
                            event_names=(TARGET, 'no-event')))
diff_binomial = pd.concat(diff_binomial).sort_values(
    ('binomial test', 'pvalue'))
diff_binomial.to_excel(writer, 'clinic binary')
with pd.option_context('display.max_rows', len(diff_binomial)):
    display(diff_binomial)

# %% [markdown]
# ## Olink - uncontrolled

# %%
olink.loc[:, olink.isna().any()].describe()

# %%
ana_diff_olink = njab.stats.groups_comparision.diff_analysis(olink,
                                         happend,
                                         event_names=(TARGET,
                                                      'no-event')).sort_values(
                                                          ('ttest', 'p-val'))
ana_diff_olink.to_excel(writer, "olink simple")
with pd.option_context('display.max_rows', len(ana_diff_olink)):
    display(ana_diff_olink)


# %% [markdown]
# ## Olink - controlled for with clinical covariates

# %%
olink.columns.name = 'OlinkID'

# %%
clinic_ancova = [TARGET, *covar]
clinic_ancova = clinic[clinic_ancova].copy()
clinic_ancova = clinic_ancova.dropna(
)  # for now discard all rows with a missing feature
categorical_columns = clinic_ancova.columns[clinic_ancova.dtypes == 'category']
print(categorical_columns)
for categorical_column in categorical_columns:
    # only works if no NA and only binary variables!
    clinic_ancova[categorical_column] = clinic_ancova[
        categorical_column].cat.codes
clinic_ancova.describe()

# %%
ancova = njab.stats.ancova.AncovaOnlyTarget(df_proteomics=olink, df_clinic=clinic_ancova, target=TARGET, covar=covar)
ancova = ancova.ancova().sort_values('qvalue')
ancova = ancova.loc[:, "p-unc":]
ancova.columns = pd.MultiIndex.from_product([['ancova'], ancova.columns],
                                         names=('test', 'var'))
ancova.to_excel(writer, "olink controlled")
ancova.head(20)

# %%
ana_diff_olink = ana_diff_olink.join(ancova.reset_index(level=-1, drop=True))
ana_diff_olink.to_excel(writer, "olink DA")
ana_diff_olink

# %%
writer.close()

# %% [markdown]
# # PCA

# %% [markdown]
# ## Missing values handling


# %%
def info_missing(df):
    N, M = olink.shape
    msg = "{} missing features out of {} measurments, corresponding to {:.3f}%"
    msg = msg.format(df.isna().sum().sum(), N * M,
                     df.isna().sum().sum() / (N * M) * 100)
    print(msg)
    return msg


_ = info_missing(olink)

# %% [markdown]
# ## PCA on scaled data
#
# - missing values set to zero

# %%
olink_scaled = StandardScaler().fit_transform(olink).fillna(0)

PCs, pca = run_pca(olink_scaled, n_components=None)
PCs.iloc[:10, :10]

# %%
olink.columns[np.argmax(np.abs(
    pca.components_[:,
                    0]))]  # eigenvector first PCa, absolut arg max -> variable

# %%
exp_var_olink = pd.Series(
    pca.explained_variance_ratio_).to_frame('explained variance')
exp_var_olink["explained variance (cummulated)"] = exp_var_olink[
    'explained variance'].cumsum()
exp_var_olink.index.name = 'PC'
ax = exp_var_olink.plot()
fig = ax.get_figure()
src.plotting.savefig(fig, name=FOLDER / '1_PCs_distribution')

# %%
ax = seaborn.scatterplot(x=PCs.iloc[:, 0],
                         y=PCs.iloc[:, 1],
                         hue=clinic[TARGET])
fig = ax.get_figure()
src.plotting.savefig(fig, name=FOLDER / '1_PC1_vs_PC2.pdf')
