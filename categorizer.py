import numpy
import pandas
import math


def categorize_col(ds, col):
    categorized = ds.copy()
    data, categories = pandas.factorize(ds[:, col])
    categorized[:, col] = data
    return categorized, categories


def replace_nans(ds, col, value):
    ds_size = len(ds)
    __ds = ds.copy()
    for idx in range(ds_size):
        item = __ds[idx, col]
        if type(item) is str:
            continue
        if math.isnan(item):
            __ds[idx, col] = value
    return __ds
