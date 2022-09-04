import config.pandas

from pathlib import Path

STUDY_ENDDATE: str = '2022-09-04'

base_folder = 'S:/SUND-CBMR-RegH-cohorts/ProDoc'
data: str = base_folder
data_processed: str = f'{base_folder}/data/processed'
folder_reports = f'{base_folder}/reports'

fname_pkl_clinic = Path(data_processed) / 'clinic.pkl'
fname_pkl_olink = Path(data_processed) / 'olink.pkl'
fname_pkl_targets = Path(data_processed) / 'targets.pkl'

from . import clinic_data

TARGETS = [
    'dead_90', 'dead_180', 'dead_wi_90_f_infl_sample',
    'dead_wi_180_f_infl_sample', 'adm_90', 'adm_180',
    'adm_wi_90_f_infl_sample', 'adm_wi_180_f_infl_sample'
]
