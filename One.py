import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import auc, roc_auc_score
from scipy.sparse import csr_matrix
from scipy.linalg import norm

# <editor-fold desc="Basic">
train = pd.read_csv('transactions.csv')
train = train.sort_values(by = 'customer_id').reset_index(drop=True)
train.head()
test_gender = pd.read_csv('customers_gender_train.csv')
# mcc = pd.read_csv('tr_mcc_codes.csv')
# tr_types = pd.read_csv('tr_types.csv')

test_gender = test_gender.sort_values(by='customer_id').reset_index(drop=True)
target = test_gender['gender'].values
# </editor-fold>

# <editor-fold desc="Exploration">
# customers_transactions = train.customer_id.value_counts().values
# plt.plot(customers_transactions)
# plt.plot(customers_transactions[-14500:])
# customers_transactions[:10000].sum()
# customers_transactions_pd = pd.Series(customers_transactions)
# </editor-fold>


# <editor-fold desc="Data preparation">

# <editor-fold desc="Saving">
lb = LabelBinarizer(sparse_output=True)
mcc_part = lb.fit_transform(np.array(train['mcc_code']))
transactions_part = lb.fit_transform(np.array(train['tr_type']))
train['mcc_tr'] = train['mcc_code'].astype(str) + '$' + train['tr_type'].astype(str)
mcc_transaction_part = lb.fit_transform(train['mcc_tr'])
joblib.dump(mcc_part,'Data/mcc_part')
joblib.dump(transactions_part,'Data/transactions_part')
joblib.dump(mcc_transaction_part,'Data/mcc_transaction_part')

# <editor-fold desc="Loading">
mcc_part = joblib.load('Data/mcc_part')
transactions_part = joblib.load('Data/transactions_part')
mcc_transaction_part = joblib.load('Data/mcc_transaction_part')
# </editor-fold>

# z = train[['customer_id','mcc_code','tr_type']].groupby(['mcc_code','tr_type'])
# t = z.agg(lambda x: tuple(x))

tmp = pd.DataFrame(train['customer_id'])
tmp['num'] = 1
tmp_agg = tmp.groupby('customer_id', sort = True).agg(np.sum)
ids = tmp_agg.index.values
nums = tmp_agg.values.ravel()
amounts = train['amount'].values#.reshape(-1,1).T

train_mask = pd.Series(ids).isin(test_gender['customer_id']).values
train_ids = ids[train_mask]
test_ids = ids[~train_mask]

def aggregate_it(data, amounts, nums, function):
    # data - data for aggregation
    # count - value of smth in a row
    # nums - first numbers of new users
    result = np.zeros((nums.shape[0],data.shape[1]))
    Num = 0
    for i in range(nums.shape[0]):
        result[i,:] = np.dot(function(amounts[Num:(Num + nums[i])]),data[Num:(Num + nums[i]),:].toarray())
        Num+=nums[i]
    return result

def create_all(p1 = True, p2 = True, p3 = True, function = lambda x: x):
    result = np.zeros((ids.shape[0],0))
    if p1: transaction = aggregate_it(transactions_part, amounts, nums, function); result = np.hstack([result,transaction])
    if p2: mcc = aggregate_it(mcc_part, amounts, nums, function); result = np.hstack([result,mcc])
    if p3: mcc_transaction = aggregate_it(mcc_transaction_part, amounts, nums, function); result = np.hstack([result,mcc_transaction])
    return result

p1,p2,p3 = False, True, False
all_0 = create_all(p1,p2,p3) # Sum
all_1 = create_all(p1,p2,p3,np.abs) # L1 sum
# all_2 = create_all(p1,p2,p3,lambda x: x**2) # L2 sum
all_count = create_all(p1,p2,p3,lambda x: x**0) # All count

df = pd.DataFrame(all_count)
corr_data = df.corr()
mask = np.zeros_like(corr_data, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_data, mask=mask, cmap=cmap, vmax=1,vmin=-1,
            square=True, xticklabels=False, yticklabels=False,
            linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation plot', fontsize=15)
# plt.yticks(np.arange(students.shape[1]))
plt.tight_layout()

corr_matrix = corr_data.copy()
corr_matrix[corr_matrix < 0.3] = 0
corr_matrix = corr_matrix.values
corr_matrix/=corr_matrix.sum(axis=0)
corr_matrix.sum(axis=1)

all_count_new = np.dot(all_count,corr_matrix)
all_count_new *= (all_count.sum(axis=1)/all_count_new.sum(axis=1)).reshape(-1,1)

(all_count_new!=0).sum()
(all_count!=0).sum()



all_once = all_count.copy()
all_once[all_once > 1] = 1 # 1 if at least one payment

def only_pos(x):
    y = x.copy()
    y[y<0.] = 0.
    return y

def only_neg(x):
    y = x.copy()
    y[y>0.] = 0.
    return y

all_pos = create_all(p1,p2,p3,only_pos) # counts only pos amounts
all_neg = create_all(p1,p2,p3,only_neg) # counts only neg amounts

sum_pos = all_pos.sum(axis=1).reshape(-1,1)
sum_neg = all_neg.sum(axis=1).reshape(-1,1)
sum_all = sum_pos + sum_neg
sum_abs = sum_pos - sum_neg

max_amount = train[['customer_id','amount']].groupby('customer_id').agg(np.max).values
min_amount = train[['customer_id','amount']].groupby('customer_id').agg(np.min).values
max_amount[max_amount<0] = 0
min_amount[min_amount>0] = 0

all_addition = np.hstack([sum_pos,sum_neg,sum_all,sum_abs,max_amount,min_amount])

# <editor-fold desc="Saving">
# joblib.dump(all_0,'Data/all_0')
# joblib.dump(all_1,'Data/all_1')
# joblib.dump(all_count,'Data/all_count')
# joblib.dump(all_once,'Data/all_once')
# joblib.dump(all_pos,'Data/all_pos')
# joblib.dump(all_neg,'Data/all_neg')
# joblib.dump(all_addition,'Data/all_addition')
# joblib.dump(target,'Data/target')
# joblib.dump(train_mask,'Data/train_mask')
# </editor-fold>

# <editor-fold desc="Loading">
all_0 = joblib.load('Data/all_0')
all_1 = joblib.load('Data/all_1')
all_count = joblib.load('Data/all_count')
all_once = joblib.load('Data/all_once')
all_pos = joblib.load('Data/all_pos')
all_neg = joblib.load('Data/all_neg')
all_addition = joblib.load('Data/all_addition')
target = joblib.load('Data/target')
train_mask = joblib.load('Data/train_mask')
# </editor-fold>

all = np.hstack([all_0,all_1,all_count,all_once,all_pos,all_neg,all_addition])

# train_new = all[train_mask]

train_new = all_count_new[train_mask]

rstate = 5
x_train, x_holdout, y_train, y_holdout = train_test_split(train_new, target, test_size=0.23, random_state=rstate)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0, random_state=rstate)

dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dholdout = xgb.DMatrix(x_holdout, y_holdout)
watchlist = [(dval,'eval')]

params = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 1,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1,
          "eval_metric": 'auc'
          }

num_trees = 5000; n_fold = 10

xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-mean'].values[-1]







gbm = xgb.train(params, dtrain, num_trees, evals = watchlist, early_stopping_rounds = 5, verbose_eval = False)

holdout_predict = gbm.predict(dholdout)
val_predict = gbm.predict(dval)

print(roc_auc_score(y_val, val_predict))
print(roc_auc_score(y_holdout, holdout_predict))



# </editor-fold>



# </editor-fold>

# train['mcc_tr'] = train['mcc_code'].apply(lambda x: str(x)+'$')

days = train.tr_datetime.apply(lambda x: x[:-9]).values.astype(int)
time = train.tr_datetime.apply(lambda x: x[-8:])
hours = time.apply(lambda x: x[:2]).values.astype(int)
minutes = time.apply(lambda x: x[3:5]).values.astype(int)
seconds = time.apply(lambda x: x[6:8])
seconds = seconds.apply(lambda x: 0 if x=='60' else x).values.astype(int)

timestamp = days * 24 * 60 + hours * 60 + minutes
daytime = (hours * 60 + minutes)/(24*60)

def cycle_it(data_raw, min, max, n, name):
    if n == 1: return data_raw
    data = np.array([data_raw]*n).T
    names = np.zeros(n).astype('object'); names[0] = name
    for k in range(1,n):
        bound = float(min )+ (float(max)-float(min))*k/n
        # data[data_raw > bound,k]+=max# + data[data_raw > bound,k]
        data[data_raw <= bound,k]+=max
        names[k] = name + '_' + str(round(bound,2))
    return pd.DataFrame(data, columns=names)

daytimes = cycle_it(daytime,0,1,3,'time').values

sns.distplot(hours,bins=24)
sns.distplot(minutes,bins = 60)
sns.distplot(seconds,bins=60)
sns.distplot(days[(days>=153)&(days<167)],bins=14)
sns.distplot(timestamp,bins=400)


zero_mask = (hours==0) & (minutes == 0) & (seconds == 0)
zero_mask.sum()
train_zero = train.loc[zero_mask]

# lr = Lasso(max_iter=1000)
#
# lr.fit(x_train,y_train)
# lr_predict = lr.predict(x_val)
#
# roc_auc_score(y_val,lr_predict)

