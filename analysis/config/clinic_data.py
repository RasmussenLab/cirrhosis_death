"""Variable groups of processed clinical data."""
vars_binary = [
    # Demographics
    'Sex',
    # Etiology of disease
    'EtiAlco',
    'EtiFat',
    'EtiHBV',
    'EtiHCV',
    'EtiPBC',
    'EtiAIH',
    'EtiMTX',
    'EtiOther',
    'EtiUnknown',
    'DecomensatedAtDiagnosis',
    # clinical variables
    'Ascites',
    'EsoBleeding',
    'HRS',
    'HE',
    'Icterus',
    'SBP',
    # 'Heartdisease',
    'Hypertension',
    'HighCholesterol',
    'Cancer',
    'Depression',
    'Psychiatric',
    'Diabetes',
    # 'IschemicHeart',
    # 'HeartFailure',
    # 'Arrythmia',
    # 'OtherHeart',
    # Medication
    'InsulinDependent',
    'Statins',
    'NonselectBetaBlock',
    # Admisson
]

vars_binary_created = [
    # from DiagnosisPlace
    'Admission',
    'AdmissionForOtherDisease',
    'Outpatient',
    # from MartialStatus
    'MaritalStatus_Divorced',
    'MaritalStatus_Married',
    'MaritalStatus_Relationship',
    'MaritalStatus_Separated',
    'MaritalStatus_Unmarried',
    'MaritalStatus_Widow/widower',
    # CauseOfDeath
    'CoD_LiverRelated',
    'CoD_NonLiver',
    'CoD_Unknown',
    # Non alcohol etiology
    'EtiNonAlco',
    # only encoded:
    # 'LiverRelated1admFromInclu',
    'HeartDiseaseTotal',  # encoded
]

# HbA1c # laboratory marker?
# H
# LiverRelated1admFromInclu

vars_cont = [
    # # Demographics
    'Age',
    # # laboratory markers
    # 'IgM',
    # 'IgG',
    # 'IgA',
    'Hgb',
    'Leucocytes',
    'Platelets',
    'Bilirubin',  # part of MELD
    'Albumin',
    'CRP',
    # 'pp', # should be the same as INR
    'INR',  # part of MELD
    'ALAT',
    'Creatinine',  # used as measure of kidney function, part of MELD
    # # Intervals
    # 'DaysToDeath',
    # 'DaysToAdmFromDiagnose',
    # 'DaysToAdmFromSample',
    # 'DaysToDeathFromDiagnose',
    # "DaysFromInclToInflSample",
    # # derived scores
    'MELD-score',
    'MELD-Na',
    'ChildPugh',
]


counts = [  # # admission
            'AmountLiverRelatedAdm',
]

# relevant comorbidities to control for (all binary)
comorbidities = [
    # 'Cancer', # unknown in CirkaFlow
    # 'Depression', # unknown in CirkaFlow
    # 'Psychiatric', # unknown in CirkaFlow
    'Diabetes',
    # 'HeartDiseaseTotal', # excluded in CirkaFlow
    'Hypertension',
    'HighCholesterol',
]

clinical_feat = [
    'Sex', 'EtiAlco', 'Ascites', 'EsoBleeding', 'Potassium',
    'InsulinDependent', 'NonselectBetaBlock', 'HeartDiseaseTotal',
    *comorbidities,
    *vars_cont
]

covars = ["Sex", "Age", "MELD-score", *comorbidities]


scores = ['MELD-score', 'MELD-Na', 'ChildPugh',]

# clini
