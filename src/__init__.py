# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
import src.plotting
from .plotting import savefig

# put into some pandas_cfg.py file and import all
import pandas as pd
import pandas.io.formats.format as pf


class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


pd.options.display.float_format = '{:,.3f}'.format
pf.IntArrayFormatter = IntArrayFormatter

plotting.make_large_descriptors()


# Load config
# from hydra import initialize_config_module, compose
# with initialize_config_module(version_base=None, config_module="config"):
#     cfg = compose('config')
#     print(cfg)
