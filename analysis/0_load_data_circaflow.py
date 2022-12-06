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
# # CircaFlow Clinical Data

# %%
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns

from lifelines import KaplanMeierFitter

import njab.plotting
from njab.plotting import savefig
import src

import config

# %%
DATA_FOLDER = Path(config.data)
DATA_PROCESSED = Path(config.data_processed)
FOLDER_REPORTS = Path(config.folder_reports) / 'data_cirkaflow'
FOLDER_REPORTS.mkdir(parents=True, exist_ok=True)

files_out=dict()

config.STUDY_ENDDATE

# %%
DATA_CLINIC = DATA_FOLDER / 'CleanData, CirKaFlow.true.xlsx'
DATA_OLINK = DATA_FOLDER / 'olink_cflow.pkl'
DATA_KEYS = DATA_FOLDER / "Validation Results" / "boks_placement_randomized.csv"
DATA_KEYS_UPDATE = DATA_FOLDER /  "Validation Results" / 'cflow_id_update.xlsx'


# %%
clinic = pd.read_excel(DATA_CLINIC).dropna(how='all', axis=1)
olink = pd.read_pickle(DATA_OLINK)

# %%
clinic = clinic.rename(config.circaflow_data.rename_dict, axis=1)
# clinic = clinic.set_index('ID')
print('Rename:',
{k: v for k, v in config.circaflow_data.rename_dict.items() if k!=v}
)

# %% [markdown]
# Find overlapping samples between keys, olink and clinic

# %%
sample_keys = pd.read_csv(DATA_KEYS, sep=';', index_col='ID')
idx_not_unique = sample_keys.index.value_counts() > 1
idx_not_unique = idx_not_unique.loc[idx_not_unique].index
duplicates = sample_keys.loc[idx_not_unique]
duplicates

# %%
sample_keys.loc[duplicates.index.unique()]

# %%
sample_keys_update = pd.read_excel(DATA_KEYS_UPDATE, index_col=0)
sample_keys_update

# %%
rename_id = sample_keys_update.dropna()['ID'].to_dict()
sample_keys = sample_keys.rename(rename_id)

# %%
clinic.insert(2, "Projekt ID", clinic.pop("Projekt ID"))
id_cols = ['SampleID', 'ID', 'Projekt ID']
clinic[id_cols] = clinic[id_cols].astype(str)
clinic.sample(3)

# %%
mask_all_equal = (clinic['SampleID'] == clinic['ID'] ) & (clinic['ID'] == clinic['Projekt ID'])
clinic.loc[~mask_all_equal]

# %%
olink.sample(3)

# %%
in_olink = olink.index.intersection(sample_keys['SampleID'])
in_olink

# %%
in_clinic = sample_keys.index.intersection(clinic['ID'])
in_clinic

# %%
assert set(in_clinic) == set(in_clinic.union(sample_keys.index.intersection(clinic['SampleID'])))

# %%
to_add = sample_keys.index.intersection(clinic['Projekt ID']).difference(in_clinic)
to_add

# %% [markdown]
# Replace `ID` with machting `Projekt ID`, then add to `in_clinc`

# %%
mask = clinic['Projekt ID'].isin(to_add)
clinic.loc[mask, 'ID'] = clinic.loc[mask, 'Projekt ID']

in_clinic = in_clinic.union(to_add)

# %%
clinic = clinic.set_index('ID')

# %%
diff_clinic_olink = clinic.index.difference(sample_keys.index)
diff_olink_clinic = sample_keys.index.difference(clinic.index)
len(in_clinic), len(diff_clinic_olink)

# %%
files_out['diff_clinic_olink'] = FOLDER_REPORTS / 'diff_clinic_olink.xlsx'
clinic.loc[diff_clinic_olink].to_excel(files_out['diff_clinic_olink'])
clinic.loc[diff_clinic_olink]


# %%
cflow_to_find = sample_keys.index.difference(in_clinic)
cflow_to_find = sample_keys.loc[cflow_to_find, 'SampleID']
cflow_to_find = cflow_to_find.loc[cflow_to_find.squeeze().str.contains('Cflow')]
if not cflow_to_find.empty:
    files_out['cflow_to_find'] = FOLDER_REPORTS / 'cflow_to_find.xlsx'
    display(files_out['cflow_to_find'])
    cflow_to_find.to_excel(files_out['cflow_to_find'])
cflow_to_find

# %% [markdown]
# Remove duplicates, i.e. if a clincal samples has more than one OLink sample (as highlighted above)

# %%
in_both = sample_keys.loc[in_clinic, 'SampleID'].index.drop_duplicates(keep=False)
len(in_both)

# %%
in_both

# %%
clinic = clinic.loc[in_both]
olink = olink.loc[sample_keys.loc[in_both, 'SampleID']]
clinic.shape, olink.shape

# %% [markdown]
# Set `SampleID` as new index for clinical data

# %%
clinic['CflowID'] = sample_keys.loc[in_both, 'SampleID']
clinic = clinic.set_index('CflowID')

# %%
# clinic[config.clinic_data.vars_binary]
vars_binary = src.pandas.get_overlapping_columns(clinic, config.clinic_data.vars_binary)
clinic[vars_binary]

# %%
vars_cont = src.pandas.get_overlapping_columns(clinic, config.clinic_data.vars_cont)
clinic[vars_cont]

# %%
clinic.dtypes.value_counts()

# %%
clinic.loc[:,clinic.dtypes == 'object']

# %%
clinic["cirrose ætiologi. Alkohol = 1, 2 = HCV, 3 = cryptogen, 4= NASH, 5= anit1-trypsin mangel, 6 hæmokromatose, 7=autoimmun og PBC, 8=HBV, 9 kutan porfyri"].value_counts()

# %% [markdown]
# Drop duplicate columns

# %%
if not clinic.columns.is_unique:
    print(f"Duplicated: {clinic.columns[clinic.columns.duplicated()]}")
    keep = ~clinic.columns.duplicated()
    clinic = clinic.loc[:, keep]
clinic.sample(3)

# %% [markdown]
# ## Discard healthy samples

# %%
clinic["Healthy"].value_counts(dropna=False)

# %%
mask = clinic["Healthy"] == 'No'
clinic, olink = clinic.loc[mask], olink.loc[mask]
clinic.shape, olink.shape

# %%
mask = clinic["DateInflSample"].isna()
clinic.loc[mask] if mask.sum() else "All included samples have an inflammation sample date"

# %% [markdown]
# ## DeathHealthyer time
#
# - one plot with absolute time axis
# - one plot relative to diagnosis date


# %%
clinic['dead'] = (clinic['DateDeath'] - clinic['DateInflSample']).notna()
clinic["DateDeath"] = clinic["DateDeath"].fillna(value=config.STUDY_ENDDATE)


# %%
din_a4 = (8.27 * 2, 11.69 * 2)
njab.plotting.make_large_descriptors(32)

fig, ax = plt.subplots(figsize=din_a4)
src.plotting.plot_lifelines(clinic.sort_values('DateInflSample'), start_col='DateInflSample', ax=ax)
_ = plt.xticks(rotation=45)
ax.invert_yaxis()
njab.plotting.set_font_sizes('x-small')
files_out['lifelines'] = FOLDER_REPORTS/ 'lifelines.pdf'
savefig(fig, files_out['lifelines'])
# %%
clinic.dead.value_counts()

# %%
fig, axes = plt.subplots(2, sharex=True)
ax =  axes[0]
ax.set_yticks([])

ax = clinic.loc[clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateInflSample", y="dead", c='blue', rot=45, ax=ax, ylabel='dead')
ax =  axes[1]
# ax.axes.yaxis.set_visible(False)
ax.set_yticks([])
ax = clinic.loc[~clinic.dead].astype({
    'dead': 'category'
}).plot.scatter(x="DateInflSample", y="dead", c='blue', rot=45, ax=ax, ylabel='alive')
_ = fig.suptitle("Inclusion date by survival status")
files_out['death_vs_alive_diagonose_dates'] = FOLDER_REPORTS / 'death_vs_alive_diagonose_dates'
savefig(fig, files_out['death_vs_alive_diagonose_dates'])


# %%
ax = clinic.astype({
    'dead': 'category'
}).plot.scatter(x="DateInflSample", y='DateDeath', c="dead", cmap='Paired', rot=45, s=3, sharex=False)
# ticks = ax.get_xticks()
# ax.set_xticklabels(ax.get_xticklabels(),  horizontalalignment='right')
# ax.set_xticks(ticks)
fontsize = 'xx-small'
min_date, max_date = clinic["DateInflSample"].min(), clinic["DateInflSample"].max()
ax.plot([min_date, max_date],
        [min_date, max_date],
        'k-', lw=1)
_ = ax.annotate('date', [min_date, min_date + datetime.timedelta(days=20)], fontsize=fontsize, rotation=25)
offset, rot = 20 , 25
delta=90
_ = ax.plot([min_date, max_date],
        [min_date + datetime.timedelta(days=delta), max_date+ datetime.timedelta(days=delta)],
        'k-', lw=1)
_ = ax.annotate(f'+ {delta} days', [min_date, min_date + datetime.timedelta(days=delta+20)], fontsize=fontsize, rotation=25)
delta=180
ax.plot([min_date, max_date],
        [min_date + datetime.timedelta(days=delta), max_date+ datetime.timedelta(days=delta)],
        'k-', lw=1)
_ = ax.annotate(f'+ {delta} days', [min_date, min_date + datetime.timedelta(days=delta+20)], fontsize=fontsize, rotation=25)
delta=360
ax.plot([min_date, max_date],
        [min_date + datetime.timedelta(days=delta), max_date+ datetime.timedelta(days=delta)],
        'k-', lw=1)
_ = ax.annotate(f'+ {delta} days', [min_date, min_date + datetime.timedelta(days=delta+20)], fontsize=fontsize, rotation=25)
fig = ax.get_figure()
files_out['timing_deaths_over_time'] = FOLDER_REPORTS / 'timing_deaths_over_time.pdf'
savefig(fig, files_out['timing_deaths_over_time'])

# %% [markdown]
# ## Cleanup steps

# %% [markdown]
# ### Clinic
#
# - [x] encode binary variables (yes, no) as `category`s
#   > Be aware that this might cause unexpected behaviour!

# %% [markdown]
# Fill derived variables with missing measurements

# %%
clinic.loc[:, clinic.columns.str.contains("Adm")].describe()

# %%
clinic.loc[:, clinic.columns.str.contains("Adm")].sum()

# %% [markdown]
# Encode binary variables

# %%
clinic[vars_binary].head()

# %%
clinic[vars_binary] = clinic[vars_binary].astype('category')


# %% [markdown]
# remaining non numeric variables

# %%
mask_cols_obj = clinic.dtypes == 'object'
clinic.loc[:,mask_cols_obj].describe()


# %%
# clinic["HbA1c"] = clinic["HbA1c"].replace(to_replace="(NA)", value=np.nan).astype(pd.Int32Dtype())
# clinic["MaritalStatus"] = clinic["MaritalStatus"].astype('category')
clinic["HeartDiseaseTotal"] = 0 # excluded from CirkaFlow
clinic["HeartDiseaseTotal"] = clinic["HeartDiseaseTotal"].replace(0, 'no').astype('category')


# %%
def get_dummies_yes_no(s, prefix=None):
    return pd.get_dummies(s, prefix=prefix).replace({
        0: 'No',
        1: 'Yes'
    }).astype('category')

# clinic = clinic.join(get_dummies_yes_no(clinic["DiagnosisPlace"]))
# clinic = clinic.join(get_dummies_yes_no(clinic["MaritalStatus"], prefix='MaritalStatus'))
clinic = clinic.join(get_dummies_yes_no(clinic["CauseOfDeath"], prefix='CoD'))
clinic

# %% [markdown]
# - few have more than one etiology

# %%
etiology_mask_yes = clinic.loc[:, clinic.columns.str.contains("Eti")] == 'Yes'
etiology_mask_yes.sum(axis=1).value_counts()

# %%
clinic["EtiNonAlco"] = (clinic["EtiAlco"] == 'No') & (etiology_mask_yes.drop('EtiAlco', axis=1).sum(axis=1).astype(bool))
clinic["EtiNonAlco"] = get_dummies_yes_no(clinic["EtiNonAlco"])[True]
clinic["EtiNonAlco"].value_counts()
# %% [markdown]
# ### Olink
#
# - [x] highlight missing values
#

# %%
olink.head()

# %% [markdown]
# Which measurments have missing values
#
# - [ ] Imputation due to limit of detection (LOD) -> how to best impute

# %%
olink.loc[:, olink.isna().any()].describe()
# %% [markdown]
# ## Timespans
#
# - death only has right censoring, no drop-out
# - admission has right censoring, and a few drop-outs who die before their first admission for the cirrhosis
#
# First some cleaning


# %%
clinic["DateAdm"]  = clinic["DateAdm"].replace({'None': np.nan, 'MORS': np.nan})


# %% [markdown]
# For these individuals, the admission time is censored as the persons died before.

# %%
clinic["DaysToAdmFromInflSample"] = (
    clinic["DateAdm"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInflSample"]).dt.days
clinic["DaysToDeathFromInfl"] = (
    clinic["DateDeath"].fillna(config.STUDY_ENDDATE) -
    clinic["DateInflSample"]).dt.days

cols_clinic = src.pandas.get_colums_accessor(clinic)

cols_view = [
    cols_clinic.DaysToDeathFromInfl,
    cols_clinic.DateAdm,
    # cols_clinic.DateFirstAdmission,
    cols_clinic.DaysToAdmFromInflSample,
    "dead",
    "Adm90", "Adm180",
    "Age"
]
# %%
mask = (clinic.dead == True) & (clinic.Adm180.isna())
view = clinic.loc[mask, cols_view].sort_values(cols_view)
files_out['died_before_admission'] = FOLDER_REPORTS / 'died_before_adm.xlsx'
view.to_excel(files_out['died_before_admission'])
view

# %% [markdown]
# ## Kaplan-Meier survival plot

# %%
kmf = KaplanMeierFitter()
kmf.fit(clinic["DaysToDeathFromInfl"], event_observed=clinic["dead"])

fig, ax = plt.subplots()
y_lim = (0, 1)
ax = kmf.plot(  #title='Kaplan Meier survival curve since inclusion',
    xlim=(0, None),
    ylim=y_lim,
    xlabel='Days since inflammation sample',
    ylabel='survival rate',
    ax=ax,
    legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
files_out['km_plot_death'] = FOLDER_REPORTS / 'km_plot_death.pdf'
savefig(fig, files_out['km_plot_death'])

# %%
clinic['Cause of Death (labels)'] = clinic['CauseOfDeath'].fillna('NA')
_ = sns.catplot(x="DaysToDeathFromInfl",
                y="dead",
                # hue="DiagnosisPlace",
                hue='Cause of Death (labels)',
                data=clinic.astype({'dead': 'category'}),
                # height=4,
                # aspect=3
               )
_.set_xlabels('Days from inflammation sample to death or until study end')
ax = _.fig.get_axes()[0]
ylim = ax.get_ylim()
ax.vlines(90, *ylim)
ax.vlines(180, *ylim)
fig = ax.get_figure()
files_out['deaths_along_time'] = FOLDER_REPORTS / 'deaths_along_time.pdf'
savefig(fig, files_out['deaths_along_time'])


# %% [markdown]
# ## KP plot admissions
#
# - some die before they have a first admission. We exclude these here

# %%
clinic["LiverAdm180"].value_counts(dropna=False).sort_index()

# %%
kmf = KaplanMeierFitter()

mask = clinic["LiverAdm180"].notna()
print(f"Based on {mask.sum()} patients")
# kmf.fit(clinic["DaysToDeathFromInfl"], event_observed=clinic["LiverAdm180"].fillna(0))
kmf.fit(clinic.loc[mask, "DaysToDeathFromInfl"], event_observed=clinic.loc[mask, "LiverAdm180"])


fig, ax = plt.subplots()
y_lim = (0, 1)
ax = kmf.plot(#title='Kaplan Meier curve for liver related admissions',
              xlim=(0, None),
              ylim=(0, 1),
              xlabel='Days since inflammation sample',
              ylabel='remaining with non-liver related admission',
              legend=False)
_ = ax.vlines(90, *y_lim)
_ = ax.vlines(180, *y_lim)
fig = ax.get_figure()
files_out['km_plot_admission'] = FOLDER_REPORTS / 'km_plot_admission.pdf'
savefig(fig, files_out['km_plot_admission'])

# %% [markdown]
# ## Targets

# %%
mask = clinic.columns.str.contains("(90|180)")
clinic.loc[:,mask] = clinic.loc[:,mask].fillna(0)
clinic.loc[:,mask].describe()

# %%
mask = clinic.columns.str.contains("Adm(90|180)")
clinic.loc[:,mask].describe() # four targets for liver related admissions

# %%
target_name = {k:f'has{k}' for k in clinic.columns[mask]}
target_name

# %%
targets = {}

for cutoff in [90, 180]:
    targets[f'dead{cutoff:03}infl'] = (clinic["DaysToDeathFromInfl"] <=
                                       cutoff).astype(int)
    targets[f"liverDead{cutoff:03}infl"] = (
        clinic.loc[clinic["CauseOfDeath"] != 'NonLiver',
                   "DaysToDeathFromInfl"] <= cutoff).astype(int)

targets = pd.DataFrame(targets)
targets = targets.join(
    (clinic.loc[:, mask] > 0).astype(int).rename(columns=target_name))
# targets = targets.sort_index(axis=1, ascending=False)
targets.describe()

# %%
src.pandas.combine_value_counts(targets)


# %% [markdown]
# add to clinical targets

# %%
clinic = clinic.join(targets)

# %% [markdown]
# ## Censoring

# %% [markdown]
# Date of first Admission is also right-censored

# %%
time_from_inclusion_to_first_admission = clinic["DateAdm"].fillna(config.STUDY_ENDDATE) - clinic["DateInflSample"]
time_from_inclusion_to_first_admission.describe()

# %% [markdown]
# Who dies without having a first Admission date?

# %%
dead_wo_adm = clinic["DateAdm"].isna() & clinic['dead']
idx_dead_wo_adm = dead_wo_adm.loc[dead_wo_adm].index
print('Dead without admission to hospital:', *dead_wo_adm.loc[dead_wo_adm].index)
clinic.loc[dead_wo_adm, ["DateAdm", "DateInflSample", cols_clinic.LiverAdm180, "CauseOfDeath"]]

# %%
DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
files_out[config.fname_pkl_cirkaflow_clinic.stem] = config.fname_pkl_cirkaflow_clinic
files_out[config.fname_pkl_cirkaflow_olink.stem] = config.fname_pkl_cirkaflow_olink
clinic.to_pickle(config.fname_pkl_cirkaflow_clinic)
olink.to_pickle(config.fname_pkl_cirkaflow_olink)
# %%
files_out
