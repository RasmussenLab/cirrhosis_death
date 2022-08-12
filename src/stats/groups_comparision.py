import pandas as pd
import pingouin as pg


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