# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
import src.io
import src.plotting
import src.pandas

# put into some pandas_cfg.py file and import all
import pandas as pd
import pandas.io.formats.format as pf

import sklearn.model_selection

class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


pd.options.display.float_format = '{:,.3f}'.format
pf.IntArrayFormatter = IntArrayFormatter



# Load config
# from hydra import initialize_config_module, compose
# with initialize_config_module(version_base=None, config_module="config"):
#     cfg = compose('config')
#     print(cfg)


def join(l): return ','.join([str(x) for x in l])


def find_val_ids(df:pd.DataFrame, val_ids:str=None, val_ids_query:str=None, sep=',') -> list:
    """Find validation IDs based on query or split."""
    if not val_ids:
        if val_ids_query:
            logging.warning(f"Querying index using: {val_ids_query}")
            val_ids = df.filter(like='Cflow', axis=0).index.to_list()
            logging.warning(f"Found {len(val_ids)} Test-IDs")
        else:
            raise ValueError("Provide a query string.")
    elif isinstance(val_ids, str):
        val_ids = val_ids.split(sep)
    else:
        raise ValueError("Provide IDs in csv format as str: 'ID1,ID2'")
    return val_ids