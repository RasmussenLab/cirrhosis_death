# put into some pandas_cfg.py file and import all
import pandas as pd
import pandas.io.formats.format as pf

pd.options.display.max_columns = 100
pd.options.display.max_rows = 30
pd.options.display.min_rows = 20
pd.options.display.float_format = '{:,.3f}'.format


class IntArrayFormatter(pf.GenericArrayFormatter):
    def _format_strings(self):
        formatter = self.formatter or '{:,d}'.format
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


pf.IntArrayFormatter = IntArrayFormatter
