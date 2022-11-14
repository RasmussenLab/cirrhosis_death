import logging

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels
from scipy.stats import binomtest as scipy_binomtest

logger = logging.getLogger(__name__)

def means_between_groups(
    df: pd.DataFrame,
    boolean_array: pd.Series,
    event_names: tuple[str, str] = ('1', '0')
) -> pd.DataFrame:
    """Mean comparison between groups"""
    sub = df.loc[boolean_array].describe().iloc[:3]
    sub['event'] = event_names[0]
    sub = sub.set_index('event', append=True).swaplevel()
    ret = sub
    sub = df.loc[~boolean_array].describe().iloc[:3]
    sub['event'] = event_names[1]
    sub = sub.set_index('event', append=True).swaplevel()
    ret = pd.concat([ret, sub])
    ret.columns.name = 'variable'
    ret.index.names = ('event', 'stats')
    return ret.T


def calc_stats(df: pd.DataFrame, boolean_array: pd.Series,
               vars: list[str]) -> pd.DataFrame:
    ret = []
    for var in vars:
        _ = pg.ttest(df.loc[boolean_array, var], df.loc[~boolean_array, var])
        ret.append(_)
    ret = pd.concat(ret)
    ret = ret.set_index(vars)
    ret.columns.name = 'ttest'
    ret.columns = pd.MultiIndex.from_product([['ttest'], ret.columns],
                                             names=('test', 'var'))
    return ret


def diff_analysis(
        df: pd.DataFrame,
        boolean_array: pd.Series,
        event_names: tuple[str, str] = ('1', '0'),
        ttest_vars=["alternative", "p-val", "cohen-d"]) -> pd.DataFrame:
    ret = means_between_groups(df,
                               boolean_array=boolean_array,
                               event_names=event_names)
    ttests = calc_stats(df, boolean_array=boolean_array, vars=ret.index)
    ret = ret.join(ttests.loc[:, pd.IndexSlice[:, ttest_vars]])
    return ret


def binomtest(var: pd.Series,
              boolean_array: pd.Series,
              alternative='two-sided',
              event_names: tuple[str, str] = ('event', 'no-event')) -> pd.DataFrame:
    entry = {}
    entry['variable'] = var.name

    assert len(
        var.cat.categories
    ) == 2, f"No binary variable, found {len(var.cat.categories)} categories: {list(var.cat.categories)}"

    p_1 = var.loc[boolean_array].dropna().cat.codes.mean()

    p_0 = var.loc[~boolean_array].dropna().cat.codes.mean()
    logger.debug(f"p cat==0: {p_0}, p cat==1: {p_1}")

    cat_at_pos_one = var.cat.categories[1]
    logger.debug('Category with code 1', cat_at_pos_one)

    counts = var.loc[boolean_array].value_counts()
    k, n = counts.loc[cat_at_pos_one], counts.sum()

    entry[event_names[0]] = dict(count=n, p=p_1)
    entry[event_names[1]] = dict(
        count=var.loc[~boolean_array].value_counts().sum(), p=p_0)

    test_res = scipy_binomtest(k, n, p_0, alternative=alternative)
    test_res = pd.Series(test_res.__dict__).to_frame('binomial test').unstack()
    test_res.name = entry['variable']
    test_res = test_res.to_frame().T

    entry = pd.DataFrame(entry).set_index('variable', append=True).unstack(0)
    entry = entry.join(test_res)
    return entry


def ancova_pg(df_long: pd.DataFrame,
              feat_col: str,
              dv: str,
              between: str,
              covar: list[str] | str,
              fdr=0.05) -> pd.DataFrame:
    """ Analysis of covariance (ANCOVA) using pg.ancova
    https://pingouin-stats.org/generated/pingouin.ancova.html
    
    Adds multiple hypothesis testing correction by Benjamini-Hochberg
    (qvalue, rejected)

    Parameters
    ----------
    df_long : pd.DataFrame
        should be long data format
    feat_col : str
        feature column (or index) name
    dv : str
        Name of column containing the dependant variable, passed to pg.ancova
    between : str
        Name of column containing the between factor, passed to pg.ancova
    covar : list, str
        Name(s) of column(s) containing the covariate, passed to pg.ancova
    fdr : float, optional
        FDR treshold to apply. , by default 0.05
    
   , long data format,
     multiple hypothesis testing corrected by Benjamini-Hochberg.
    Note that column name of dv shouldn't contain '\t'
    "data": 
    "dv": Name of column containing the dependant variable.
    "between": Name of column containing the between factor.
    "covar": Name(s) of column(s) containing the covariate. 
    More refer to: 


    Returns
    -------
    pd.DataFrame
        Columns:  [ 'Source',
                    'SS',
                    'DF',
                    'F',
                    'p-unc',
                    'np2',
                    '{feat_col}',
                    '-Log10 pvalue',
                    'qvalue',
                    'rejected']
    """
    scores = []
    # num_covar = len(covar)

    for feat_name, data_feat in df_long.groupby(feat_col):
        # from IPython.core.debugger import set_trace; set_trace()
        ancova = pg.ancova(data=data_feat, dv=dv, between=between, covar=covar)
        ancova[feat_col] = feat_name
        scores.append(ancova)
    scores = pd.concat(scores)
    scores['-Log10 pvalue'] = -np.log10(scores['p-unc'])
    return scores


def add_fdr_scores(scores: pd.DataFrame,
                   random_seed: int = None,
                   alpha=0.05,
                   method='indep') -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)
    reject, qvalue = statsmodels.stats.multitest.fdrcorrection(scores['p-unc'],
                                                               alpha=alpha,
                                                               method=method)
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    return scores


def ancova_per_feat(df_proteomics: pd.DataFrame,
            df_clinic: pd.DataFrame,
            target: str,
            covar: list[str],
            value_name: str = 'intensity') -> pd.DataFrame:
    """apply ancova and multiple test correction. `df_proteomics` is yielding the features
    of interest, `df_clinic` is providing the covariates to be controlled for.

    Parameters
    ----------
    df_proteomics : pd.DataFrame
        proteomic measurements in wide format
    df_clinic : pd.DataFrame
        clinical data, containing `target` and `covar`
    target : str
        Variable for stratification contained in `df_clinic`
    covar : list[str]
        List of control varialbles contained in `df_clinic`
    value_name : str
        Name to be used for protemics measurements in long-format, default "intensity"

    Returns
    -------
    pd.DataFrame
        Columns = ['SS', 'DF', 'F', 'p-unc', 'np2', '-Log10 pvalue',
                   'qvalue', 'rejected']
    """

    data = (df_proteomics.loc[df_clinic[target].notna()].stack().to_frame(
        value_name).join(df_clinic))
    feat_col = data.index.names[-1]
    scores = ancova_pg(data,
                       feat_col=feat_col,
                       dv=value_name,
                       between=target,
                       covar=covar)


    scores = scores[scores.Source != 'Residual']

    #FDR correction
    scores = add_fdr_scores(scores, random_seed=123)
    return scores.set_index([feat_col, 'Source'])
