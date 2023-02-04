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
# # Olink validation data
#
# both for 
#
# - `ProDoc` (N=29) samples
# - `CircaFlow` samples

# %%
from collections import namedtuple
from pathlib import Path

import pandas as pd

import config
DATA_FOLDER = Path(config.data)
DATA_FOLDER

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
OLINK:str = DATA_FOLDER / "Validation Results" / "ProDoc_Olink_bridged_QC_long.tsv"
METADATA:str = DATA_FOLDER / "Validation Results" / "metadata.tsv"
ID_MAP:str = DATA_FOLDER / "Validation Results" / "id.xlsx"
OLINK_UPDATE:str = DATA_FOLDER / "Validation Results" / "update_olink_221204.tsv"

# %% [markdown]
# ## Set default paths and collection

# %%
inputs = {}
outputs = {}

inputs['olink'] = OLINK
inputs['metadata'] = METADATA
inputs['id_map'] = ID_MAP
inputs['olink_update'] = OLINK_UPDATE

inputs

# %% [markdown]
# ## Define Measurment

# %%
Measurement = namedtuple('Measurment', 'idx measure')
measure_olink = Measurement(['SampleID', 'Assay'], 'NPX')
measure_olink

# %% [markdown]
# # Load Olink validation data

# %%
olink = pd.read_table(inputs['olink'], sep='\t', low_memory=False)
olink = olink.set_index(measure_olink.idx)
olink

# %% [markdown]
# # Contains duplicated bridging samples

# %%
duplicated = olink[measure_olink.measure].index.duplicated(keep=False)
olink_bridge = olink.loc[duplicated].sort_index(level=-1).set_index('Project',
                                                                    append=True)
olink_bridge.head(20)

# %%
outputs['bridging_samples'] = config.data_processed / 'bridges.pkl'
olink_bridge.to_pickle(outputs['bridging_samples'])
olink_bridge.to_excel(outputs['bridging_samples'].with_suffix('.xlsx'))

# %% [markdown]
# # Metadata for Olink features
#
# - `UniProt` ID of `OlinkID`
# - limit of detection (`LOD`)

# %%
metadata = pd.read_table(inputs["metadata"])
metadata

# %% [markdown]
# # Sample name to ID mapping  - find subcohorts

# %%
id_map = pd.read_excel(inputs["id_map"], index_col='SampleID')
id_map

# %%
print(id_map["CBMRID"].str[:4].value_counts().to_string())


# %% [markdown]
# # Select cohorts

# %%
def _select_idx(query: str,
                expected: int,
                id_map: pd.DataFrame = id_map,
                id_col: str = 'CBMRID'):
    idx = id_map.loc[id_map[id_col].str.contains(query)]
    idx = idx[id_col].to_list()
    assert len(
        idx
    ) == expected, f"Excepcted {expected} Prodoc validation samples, not {len(idx)}"
    return idx


# %%
idx_prodoc = _select_idx(query='ProD', expected=29)
# idx_prodoc

# %%
idx_cirkaflow = _select_idx(query='Cflow', expected=101)
# idx_cirkaflow

# %%
olink_prodoc_val = olink.loc[idx_prodoc, measure_olink.measure].unstack()
olink_prodoc_val.describe()

# %%
stem = 'olink_prodoc_val'
outputs[f'{stem}'] = config.data_processed / f'{stem}.pkl'
olink_prodoc_val.to_pickle(outputs[f'{stem}'])
olink_prodoc_val.to_excel(outputs[f'{stem}'].with_suffix('.xlsx'))

# %%
olink_cflow = olink.loc[idx_cirkaflow, measure_olink.measure].unstack()
olink_cflow.describe()

# %% [markdown]
# Integrate update from Rasmus (last three non-matching IDs)

# %%
olink_update = pd.read_table(inputs['olink_update'], sep='\t', low_memory=False)
olink_update = olink_update.set_index(measure_olink.idx)

olink_cflow_update = olink_update.loc[:, measure_olink.measure].unstack()
olink_cflow_update

# %%
olink_cflow.loc[olink_cflow_update.index]

# %%
stem = 'olink_cflow'
outputs[stem] = config.data_processed / f'{stem}.xlsx'
olink_cflow.to_excel(outputs[stem])
olink_cflow.to_pickle(outputs[stem].with_suffix('.pkl'))

# %% [markdown]
# Log all input and selected output files 

# %%
inputs

# %%
outputs
