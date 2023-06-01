# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 18:02:44 2021

@author: begas05
"""

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

'''
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        
del(name)
'''
pd.set_option(#'max_columns', 4,
    'max_rows', 10)
from io import StringIO
def txt_repr(df, width=40, rows=None):
    buf = StringIO()
    rows = rows if rows is not None else pd.options.display.max_rows
    num_cols = len(df.columns)
    with pd.option_context('display.width', 100):
        df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
        out = buf.getvalue()
        for line in out.split('\n'):
            if len(line) > width or line.strip().endswith('\\'):
                break
        else:
            return out
        done = False
        while not done:
            buf = StringIO()
            df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
            for line in buf.getvalue().split('\n'):
                if line.strip().endswith('\\'):
                    num_cols = min([num_cols - 1, int(num_cols*.8)])
                    break
            else:
                break
        return buf.getvalue()
pd.DataFrame.__repr__ = lambda self, *args: txt_repr(self, 65, 10)





