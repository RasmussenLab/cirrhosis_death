import pandas as pd

import sklearn

from src.sklearn.pca import run_pca
from src.sklearn.preprocessing import StandardScaler


from .types import Splits, ResultsSplit, Results, AucRocCurve, PrecisionRecallCurve



def get_results_split(clf: sklearn.base.BaseEstimator, X: pd.DataFrame,
                      y: pd.Series) -> ResultsSplit:
    results = ResultsSplit(auc=clf.score(X, y))

    results.auc = clf.score(X, y)
    pred_score_target = clf.predict_proba(X)
    N, n_classes = pred_score_target.shape
    assert n_classes == 2, f"Non binary classification: {y.unique()}"
    pred_score_target = pred_score_target[:, 1]

    results.roc = AucRocCurve(
        *sklearn.metrics.roc_curve(y_true=y, y_score=pred_score_target))
    results.prc = PrecisionRecallCurve(*sklearn.metrics.precision_recall_curve(
        y_true=y, probas_pred=pred_score_target))

    results.aps = sklearn.metrics.average_precision_score(
        y_true=y, y_score=pred_score_target)
    return results


def transform_DataFrame(X:pd.DataFrame, fct):
    ret = fct(X)
    ret = pd.DataFrame(ret, index=X.index, columns=X.columns)
    return ret