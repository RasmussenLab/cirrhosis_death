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
import numpy as np
import pandas as pd

import pingouin as pg

import src.stats
from src.sklearn import run_pca, StandardScaler

import config

# %% [markdown]
# ## Set parameters

# %%
TARGET = 'dead_wi_90_f_infl_sample'

# %%
clinic = pd.read_pickle(config.fname_pkl_clinic)
olink = pd.read_pickle(config.fname_pkl_olink)

# %%
pd.crosstab(clinic.DiagnosisPlace, clinic.dead)

# %% [markdown]
# FirstAdmission is also right-censored

# %%
time_from_diagnose_to_first_admission = clinic["DateFirstAdmission"].fillna(config.STUDY_ENDDATE) - clinic["DateDiagnose"]
time_from_diagnose_to_first_admission.describe()

# %% [markdown]
# Who dies without having a first Admission date?

# %%
dead_wo_adm = clinic["DateFirstAdmission"].isna() & clinic['dead']
idx_dead_wo_adm = dead_wo_adm.loc[dead_wo_adm].index
print('Dead without admission to hospital:', *dead_wo_adm.loc[dead_wo_adm].index)
clinic.loc[dead_wo_adm, ["DateFirstAdmission", "DateDiagnose", "Admissions"]]

# %% [markdown]
# ## Differences between groups defined by target

# %%
clinic

# %%
clinic[TARGET].value_counts()

# %%
pd.crosstab(clinic[TARGET], clinic["DecomensatedAtDiagnosis"])

# %%
happend = clinic[TARGET].astype(bool)

# %% [markdown]
# ### Continous

# %%
var = 'Age'
# import scipy.stats 
# scipy.stats.ttest_ind(clinic.loc[happend, var], clinic.loc[~happend, var], equal_var=False) # same results as pengoin
pg.ttest(clinic.loc[happend, var], clinic.loc[~happend, var])

# %%
vars_cont = config.clinic_data.vars_cont
ana_differential = src.stats.diff_analysis(
    clinic[vars_cont],
    happend,
    event_names=('died', 'alive'),
)
ana_differential.sort_values(('ttest', 'p-val'))

# %% [markdown]
# ### Binary

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
      diff_binomial.append(src.stats.binomtest(clinic[var], happend))
pd.concat(diff_binomial).sort_values(('binomial test', 'pvalue'))

# %% [markdown]
# ## Olink

# %%
olink.loc[:, olink.isna().any()].describe()

# %%
ana_diff_olink = src.stats.diff_analysis(olink, happend, event_names=('died', 'alive'))
ana_diff_olink.sort_values(('ttest', 'p-val'))


# %% [markdown]
# ## PCA 

# %% [markdown]
# ### Missing values handling

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
# ### PCA on scaled data 
#
# - missing values set to zero

# %%
olink_scaled = StandardScaler().fit_transform(olink).fillna(0)

PCs, pca = run_pca(olink_scaled, n_components=None)
PCs

# %%
olink.columns[np.argmax(np.abs(pca.components_[:,0]))] # eigenvector first PCa, absolut arg max -> variable

# %%
exp_var_olink = pd.Series(pca.explained_variance_ratio_).to_frame('explained variance')
exp_var_olink["explained variance (cummulated)"] = exp_var_olink['explained variance'].cumsum()
exp_var_olink.index.name = 'PC'
ax = exp_var_olink.plot()

# %% [markdown]
# ### Logistic Regression

# %%
import sklearn
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.sklearn.scoring import ConfusionMatrix

y_true = clinic[TARGET]
X = PCs.iloc[:,:5]
y_true.value_counts()

# %% [markdown]
# #### With weights

# %%
weights= sklearn.utils.class_weight.compute_sample_weight('balanced', y_true)

log_reg = sklearn.linear_model.LogisticRegression()
log_reg = log_reg.fit(X=X, y=y_true, sample_weight=weights)

# %%
scores = dict(ref_score=(y_true.value_counts() / len(clinic)).max(),
              model_score=log_reg.score(X, y_true, sample_weight=None))

scores

# %%
y_pred = log_reg.predict(X)

ConfusionMatrix(y_true, y_pred).as_dataframe

# %%
pivot = y_true.to_frame()
pivot['pred'] = y_pred
pivot = pivot.join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %%
pd.pivot_table(pivot, values='pred', index=TARGET, columns='dead', aggfunc='sum')

# %%
pd.pivot_table(pivot, values='dead', index=TARGET, columns='pred', aggfunc='sum')

# %%
pivot.groupby(['pred', TARGET]).agg({'dead': ['count', 'sum']}) # more detailed

# %% [markdown]
# #### Without weights, but adapting cutoff

# %%
log_reg = log_reg.fit(X=X, y=y_true, sample_weight=None)

y_prob = log_reg.predict_proba(X)[:,1]
y_pred = pd.Series((y_prob > 0.21), index=PCs.index).astype(int)

ConfusionMatrix(y_true, y_pred).as_dataframe # this needs to be augmented with information if patient died by now (to see who is "wrongly classified)")

# %%
pivot = y_pred.to_frame('pred').join(y_true).join(clinic.dead.astype(int))
pivot.describe().iloc[:2]

# %% [markdown]
# How many will die for those who have been predicted to die?

# %%
pd.pivot_table(pivot, values='pred', index=TARGET, columns='dead', aggfunc='sum')

# %%
pivot.groupby(['pred', TARGET]).agg({'dead': ['count', 'sum']}) # more detailed

# %%
fpr, tpr, cutoffs = roc_curve(y_true, y_prob)
roc = pd.DataFrame([fpr, tpr, cutoffs[::-1]], index='fpr tpr cutoffs'.split())
ax = roc.T.plot('fpr', 'tpr')

# %%
precision, recall, cutoffs = precision_recall_curve(y_true, y_prob)
prc = pd.DataFrame([precision, recall, cutoffs[::-1]], index='precision recall cutoffs'.split())
prc

# %%
ax = prc.T.plot('recall', 'precision', ylabel='precision')
