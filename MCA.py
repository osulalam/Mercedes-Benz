import pandas as pd

train = pd.read_csv (
	r"C:\Users\Osula\Documents\Projects\Mercedes-Benz\train.csv")
train.shape
test = pd.read_csv (
	r"C:\Users\Osula\Documents\Projects\Mercedes-Benz\test.csv")
test.shape

# save IDs for submission
id_test = test['ID'].copy()

## Select all columns in our train dataset that only have 1 unique value
train_drop = train.loc[:,(train.apply(pd.Series.nunique) == 1)].columns.tolist()

## Select all columns in our test dataset that only have 1 unique value
test.loc[:,(test.apply(pd.Series.nunique) == 1)].columns.tolist()

## Let's drop the columns from the training dataset that are homogeneous
## throughout the training set.
print("Before shape : {}".format(train.shape))
train = train.drop(train.loc[:,(train.apply(pd.Series.nunique) == 1)].columns.tolist(), axis=1)
print("After shape : {}".format(train.shape))

## We need to drop the same columns from the test dataset.
print("Before shape : {}".format(test.shape))
test = test.drop(train_drop, axis=1)
print("After shape : {}".format(test.shape))

# glue datasets together
both = pd.concat([train, test], axis=0) # concatenate along axis = 0 (default) added below according to index
print('initial shape: {}'.format(both.shape))
both.head()
both.tail()

# binary indexes for train/test set split
is_train = ~both.y.isnull() # ~ = y is NOT null = True

# find all categorical features
cf = both.select_dtypes(include=['object']).columns
print(cf)

import mca

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = mca.dummy(
    both[cf]
)
print('oh-encoded shape: {}'.format(dummies.shape))
dummies.head()

# get rid of old columns and append them encoded
both = pd.concat(
    [
        both.drop(cf, axis=1), # drops the columns of cf (axis=1 refers to columns); 'cf' = list of column titles to drop
        dummies # append them one-hot-encoded
    ],
    axis=1 # column-wise (attached next to other set without overriding the other rows or columns) if axis=0 will join by index (below rest of data without overlapping feature values; will create columns if feature not available)
)
print('appended-encoded shape: {}'.format(both.shape))

# recreate train/test again, now with dropped ID column
train, test = both[is_train].drop(['ID'], axis=1), both[~is_train].drop(['ID', 'y'], axis=1)

mca_ben = mca.mca(train.drop(['y'], axis=1), ncols = 364)
mca_ind = mca.mca(train.drop(['y'], axis=1), ncols = 364, benzecri=False)

