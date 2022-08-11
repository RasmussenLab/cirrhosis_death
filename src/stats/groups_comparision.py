import pandas as pd

def means_between_groups(df:pd.DataFrame, bool_array:pd.Series, event_names:tuple[str, str]=('1', '0')):
    """Mean comparison between groups"""
    sub = df.loc[bool_array].describe().iloc[:3]
    sub['event'] = event_names[0]
    sub = sub.set_index('event', append=True).swaplevel()
    ret = sub
    sub = df.loc[~bool_array].describe().iloc[:3]
    sub['event'] = event_names[1]
    sub = sub.set_index('event', append=True).swaplevel()
    ret = pd.concat([ret, sub])
    ret.columns.name = 'variable'
    ret.index.names = ('event', 'stats')
    return ret.T