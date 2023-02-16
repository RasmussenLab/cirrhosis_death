import config.pandas

from pathlib import Path

STUDY_ENDDATE: str = '2022-09-09'
MAX_DAYS_INTERVAL: int = 730

base_folder = Path('S:/SUND-CBMR-RegH-cohorts/ProDoc')
data = Path(base_folder)
data_processed = base_folder / 'data/processed'
folder_reports = base_folder / 'reports'

fname_pkl_clinic = data_processed / 'clinic.pkl'
fname_pkl_val_clinic = data_processed / 'val_clinic.pkl'
fname_pkl_olink = data_processed / 'olink.pkl'

fname_pkl_prodoc_olink = data_processed / 'prodoc_olink_all.pkl'
fname_pkl_prodoc_clinic = data_processed / 'prodoc_clinic_all.pkl'  # with categorical dtype
fname_pkl_prodoc_clinic_num = data_processed / 'prodoc_clinic_all_numeric.pkl'  # to be used in sklearn

fname_pkl_targets = data_processed / 'targets.pkl'

fname_pkl_cirkaflow_olink = data_processed / 'cirkaflow_olink_all.pkl'
fname_pkl_cirkaflow_clinic = data_processed / 'cirkaflow_clinic_all.pkl'
fname_pkl_cirkaflow_clinic_num =  data_processed / 'cirkaflow_clinic_all_numeric.pkl'

fname_pkl_all_clinic_num = data_processed / 'all_clinic_num.pkl'
fname_pkl_all_olink = data_processed / 'all_olink.pkl'

from . import clinic_data
from . import olink
from . import cirkaflow_data

TARGETS = [
    # 'dead090infl',
    'dead180infl',  # excluded for now
    # 'liverDead090infl', 'liverDead180infl',
    # 'hasAdm180', 'hasAdm90',
    # 'hasLiverAdm90',
    'hasLiverAdm180'  # excluded for now
]

TARGET_LABELS = {'hasLiverAdm180': 'Admission',
 'dead180infl': 'Death',
 'is_valdiation_sample': 'Validation sample'}

TRAIN_LABEL = 'study group'
TEST_LABEL = 'stable cohort'

COMPARE_PRODOC = 'is_valdiation_sample'

Y_KM = {
    'dead180infl': ('DaysToDeathFromInfl', 'dead'),
    'hasLiverAdm180': ('DaysToAdmFromInflSample', 'hasLiverAdm180'),
    'is_valdiation_sample': ('DaysToDeathFromInfl', 'is_valdiation_sample')
}


feat_sets = {
    'OLINK': ','.join(olink.inflammation_panel),
    'CLINIC': ','.join(clinic_data.clinical_feat)
}
feat_sets['OLINK_AND_CLINIC'] = ','.join(feat_sets.values())
feat_sets['OLINK_AND_SCORES'] =  ','.join(olink.inflammation_panel + clinic_data.scores)
feat_sets['SCORES_ONLY'] = ','.join(clinic_data.scores)

MODEL_NAMES = {
    'OLINK': 'Inflammation markers',
    'CLINIC': 'Clinical variables',
    'SCORES_ONLY': 'Medical scores',
    'OLINK_AND_CLINIC': 'Infl. markers and clinical variables',
    'OLINK_AND_SCORES': 'Infl. markers and medical scores'
}
