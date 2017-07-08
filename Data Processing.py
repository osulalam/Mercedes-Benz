import pandas as pd

both = pd.read_csv (
	r"C:\Users\Osula\Documents\Projects\Mercedes-Benz\bothsets.csv")
both.head()

# binary indexes for train/test set split
is_train = ~both.y.isnull() # ~ = y is NOT null = True

# find all categorical features
cf = both.select_dtypes(include=['object']).columns
print(cf)

aa = ~both.X0.isin(['<211']) #insignificant at alpha = 0.05
aa = ~both.X1.isin(['<211']) #significant at alpha = 0.01
aa = ~both.X2.isin(['<211']) #significant at alpha = 0.01
aa = ~both.X4.isin(['<211']) #insignificant at alpha = 0.05
aa = ~both.X5.isin(['<211']) #significant at alpha = 0.05

from statsmodels.formula.api import ols

mod = ols('y ~ aa', data=both).fit()
mod.summary()

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    both[cf],
)
print('oh-encoded shape: {}'.format(dummies.shape))
dummies.head()

# get rid of old columns and append them encoded
both = pd.concat(
    [
        both.drop(cf, axis=1), # drops the columns of cf (axis=1 refers to columns); 'cf' = list of column titles to drop
        dummies, # append them one-hot-encoded
    ],
    axis=1 # column-wise (attached next to other set without overriding the other rows or columns) if axis=0 will join by index (below rest of data without overlapping feature values; will create columns if feature not available)
)
print('appended-encoded shape: {}'.format(both.shape))

# recreate train/test again, now with dropped ID column
train, test = both[is_train], both[~is_train]
# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

import matplotlib.pyplot as plt
import scipy.stats as sp

nf = ['X0_Counts', 'X1_Counts', 'X2_Counts', 'X4_Counts', 'X5_Counts']
for i in train[nf].columns:
    print(sp.skew(train[i]))
    print(plt.hist(train[i]))
    print(plt.title(i))

from sklearn.preprocessing import MinMaxScaler as mm
from sklearn.preprocessing import FunctionTransformer as ft
import numpy as np
from scipy.stats import boxcox as bx

train = pd.DataFrame(data=train)
mms = mm()

train = pd.DataFrame(data=train)
mms = mm()
fts = ft()
t_mm = mms.fit_transform(train[nf])
train[nf] = ft(np.log1p).transform(t_mm)
train.head()

test = pd.DataFrame(data=test)
t_mm1 = mms.fit_transform(test[nf])
test[nf] = ft(np.log1p).transform(t_mm1)
test.head()

for i in train[nf].columns:
    print (plt.hist(train[i]))
    print (plt.title(i))
    print(sp.skew(train[i]))