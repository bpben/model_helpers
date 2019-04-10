import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from ..model_helpers import Indata

def test_splitter():
    X = np.zeros(shape=(3,4))
    X = np.vstack([X, X+1, X+2])
    y = X[:,0]
    groups = X[:,0]
    X_df = pd.DataFrame(X)
    # testing index works
    X_df.index = np.arange(len(X))+1
    d = Indata(X_df, y)
    params = {'n_splits':1, 'test_size':0.3}
    # normal behavior: Split 1:3
    d.tr_te_split()
    # stratified: even number of each class
    sss = StratifiedShuffleSplit(**params)
    d.tr_te_split(splitter=sss, y=y)
    assert len(np.unique(d.test_x)) == 3
    # group: exclusive class in test/train
    gss = GroupShuffleSplit(**params)
    d.tr_te_split(splitter=gss, y=y,  groups=groups)
    assert len(np.unique(d.test_x)) == 1
    return(d)