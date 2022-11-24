import config.pandas

from pathlib import Path

STUDY_ENDDATE: str = '2022-09-09'

base_folder = 'S:/SUND-CBMR-RegH-cohorts/ProDoc'
data: str = base_folder
data_processed: str = f'{base_folder}/data/processed'
folder_reports = f'{base_folder}/reports'

fname_pkl_clinic = Path(data_processed) / 'clinic.pkl'
fname_pkl_val_clinic = Path(data_processed) / 'val_clinic.pkl'
fname_pkl_olink = Path(data_processed) / 'olink.pkl'

fname_pkl_prodoc_olink = Path(data_processed) / 'prodoc_olink_all.pkl'
fname_pkl_prodoc_clinic = Path(
    data_processed) / 'prodoc_clinic_all.pkl'  # with categorical dtype
fname_pkl_prodoc_clinic_num = Path(
    data_processed) / 'prodoc_clinic_all_numeric.pkl'  # to be used in sklearn

fname_pkl_targets = Path(data_processed) / 'targets.pkl'

from . import clinic_data
from . import olink

TARGETS = [
    # 'dead090infl',
    'dead180infl',  # excluded for now
    # 'liverDead090infl', 'liverDead180infl',
    # 'hasAdm180', 'hasAdm90',
    # 'hasLiverAdm90',
    'hasLiverAdm180'  # excluded for now
]

COMPARE_PRODOC = 'is_valdiation_sample'


feat_sets = {
    'OLINK': ','.join(olink.inflammation_panel),
    'CLINIC': ','.join(config.clinic_data.comorbidities + config.clinic_data.vars_cont)
}
feat_sets['OLINK_AND_CLINIC'] = ','.join(feat_sets.values())