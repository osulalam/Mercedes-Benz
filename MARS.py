import pandas as pd

train = pd.read_csv (
	r"C:\Users\Osula\Documents\Projects\Mercedes-Benz\train.csv")
train.head()
train.shape
test = pd.read_csv (
	r"C:\Users\Osula\Documents\Projects\Mercedes-Benz\test.csv")

# Remove the outlier
# train=train[train.y<250]

# save IDs for submission
id_test = test['ID'].copy()

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder

# glue datasets together
both = pd.concat([train, test], axis=0) # concatenate along axis = 0 (default) added below according to index
print('initial shape: {}'.format(both.shape))
both.head()
both.tail()

both.to_csv('Bothsets.csv', index=False)

# binary indexes for train/test set split
is_train = ~both.y.isnull() # ~ = y is NOT null = True

# find all categorical features
cf = both.select_dtypes(include=['object']).columns
print(cf)

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    both[cf],
    drop_first=False # you can set it = True to omit multicollinearity (crucial for linear models)
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
train, test = both[is_train], both[~is_train]
# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))
# permanently deletes the df both
del both

#Multivariate Adaptive Regression Splines
import numpy as np
import matplotlib.pyplot
import sklearn.metrics as metrics
from pyearth import Earth as earth

y = train.y
y.head()
NoY_train = train.drop(['y'], axis=1) # ~ = y is null (AKA Test Data) = FALSE
NoY_train.head()
X = NoY_train.iloc[:,1:train.shape[1]]
X.head()
xlabel = list(X.columns)
print(xlabel)
y = train.iloc[:, 1]

# Fit an Earth model
model = earth(enable_pruning = True, penalty = 3, minspan_alpha = 0.05, endspan_alpha = 0.05, feature_importance_type='rss', verbose=3)
model.fit(X, y=y, sample_weight=None, output_weight=None, xlabels = xlabel)
model.summary()

metrics.r2_score(y, model.predict(X))

NoY_test = test.drop(['y'], axis=1) # ~ = y is null (AKA Test Data) = FALSE
NoY_test.head()
Y_test = NoY_test.iloc[:,1:train.shape[1]]
Y_test.head()

res = model.predict(Y_test).ravel()

# create df and convert it to csv
output = pd.DataFrame({'ID': id_test, 'y': res})
output.to_csv('MARS_Response.csv', index=False)
