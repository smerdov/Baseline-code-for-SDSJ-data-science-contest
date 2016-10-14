import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import seaborn as sns
import scipy
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, roc_auc_score, pairwise_distances_argmin_min, pairwise_distances
from sklearn.ensemble import RandomForestClassifier
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

loading = True
n_classes = 2
# </editor-fold>

# <editor-fold desc="Exploration">
# customers_transactions = train.customer_id.value_counts().values
# plt.plot(customers_transactions)
# plt.plot(customers_transactions[-14500:])
# customers_transactions[:10000].sum()
# customers_transactions_pd = pd.Series(customers_transactions)
# </editor-fold>

# <editor-fold desc="Data preparation">

# <editor-fold desc="Binarizing">
# <editor-fold desc="Saving">
lb = LabelBinarizer(sparse_output=True)
mcc_part = lb.fit_transform(np.array(train['mcc_code']))
transactions_part = lb.fit_transform(np.array(train['tr_type']))
train['mcc_tr'] = train['mcc_code'].astype(str) + '$' + train['tr_type'].astype(str)
mcc_transaction_part = lb.fit_transform(train['mcc_tr'])

joblib.dump(mcc_part,'Data/mcc_part')
joblib.dump(transactions_part,'Data/transactions_part')
joblib.dump(mcc_transaction_part,'Data/mcc_transaction_part')
# </editor-fold>

# <editor-fold desc="Loading">
mcc_part = joblib.load('Data/mcc_part')
transactions_part = joblib.load('Data/transactions_part')
mcc_transaction_part = joblib.load('Data/mcc_transaction_part')
# </editor-fold>
# </editor-fold>

# <editor-fold desc="Preparation 1">
counter = pd.DataFrame(train['customer_id'])
counter['count'] = 1
counter_agg = counter.groupby('customer_id', sort = True).agg(np.sum)
ids = counter_agg.index.values # sorted ids of all customers
nums = counter_agg.values.ravel() # № of visits of each customer, sorted by id
amounts = train['amount'].values # amounts, sorted by id
amounts_pos = amounts.copy(); amounts_neg = amounts.copy()
amounts_pos[amounts_pos<0] = 0; amounts_neg[amounts_neg>0] = 0
amounts_pos_fact = amounts_pos.copy(); amounts_neg_fact = amounts_neg.copy()
amounts_pos_fact[amounts_pos_fact > 0] = 1; amounts_neg_fact[amounts_neg_fact < 0] = 1

train_mask = pd.Series(ids).isin(test_gender['customer_id']).values

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

def create_all(amounts, p1 = True, p2 = True, p3 = True, function = lambda x: x):
    result = np.zeros((ids.shape[0],0))
    if p1: transaction = aggregate_it(transactions_part, amounts, nums, function); result = np.hstack([result,transaction])
    if p2: mcc = aggregate_it(mcc_part, amounts, nums, function); result = np.hstack([result,mcc])
    if p3: mcc_transaction = aggregate_it(mcc_transaction_part, amounts, nums, function); result = np.hstack([result,mcc_transaction])
    return result

p1,p2,p3 = True, True, False # using of tr, mcc, tr&mcc data accordingly
all_count = create_all(amounts,p1,p2,p3,lambda x: x**0) # All count
all_pos = create_all(amounts_pos,p1,p2,p3) # Only pos amounts
all_neg = create_all(amounts_neg,p1,p2,p3) # Only neg amounts
all_pos_count = create_all(amounts_pos_fact,p1,p2,p3) # Counting pos amounts
all_neg_count = create_all(amounts_neg_fact,p1,p2,p3) # Counting neg amounts

# This is support arrays in which all 0 have replaced by 1. It doesn't change means
all_pos_count_sup = all_pos_count.copy(); all_pos_count_sup[all_pos_count_sup==0] = 1
all_neg_count_sup = all_neg_count.copy(); all_neg_count_sup[all_neg_count_sup==0] = 1

all_mean_pos = all_pos/all_pos_count_sup
all_mean_neg = all_neg/all_neg_count_sup

# Checking we are not dividing by zero
(all_mean_pos!=all_mean_pos).sum()
(all_mean_neg!=all_mean_neg).sum()

# </editor-fold>

# <editor-fold desc="Correlation">
# corr_data = pd.DataFrame(all_count).corr()

# <editor-fold desc="Graph">
# mask = np.zeros_like(corr_data, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.heatmap(corr_data, mask=mask, cmap=cmap, vmax=1,vmin=-1,
#             square=True, xticklabels=False, yticklabels=False,
#             linewidths=.5, cbar_kws={"shrink": .5})
# plt.title('Correlation plot', fontsize=15)
# # plt.yticks(np.arange(students.shape[1]))
# plt.tight_layout()
# </editor-fold>

# corr_matrix = corr_data.copy()
# corr_matrix = np.array(corr_matrix)
# #corr_matrix = corr_matrix**2
# corr_matrix[corr_matrix < 0.05] = 0
#
# all_count_new = np.dot(all_count,corr_matrix)
# all_count_new *= (all_count.sum(axis=1)/all_count_new.sum(axis=1)).reshape(-1,1)
#
# (all_count!=0).sum()
# (all_count_new!=0).sum()
# </editor-fold>

# <editor-fold desc="Other features">
max_amount = train[['customer_id','amount']].groupby('customer_id').agg(np.max).values
min_amount = train[['customer_id','amount']].groupby('customer_id').agg(np.min).values
max_amount[max_amount<0] = 0
min_amount[min_amount>0] = 0

days = train.tr_datetime.apply(lambda x: x[:-9]).values.astype(int)

# a = 0; b = 457
# a = 250; b = 300
# d2plot = days[(days>=a)&(days<b)]
# sns.distplot(d2plot, bins=b-a)
# day_zero + np.timedelta64(153,'D') # 1 Jan 2015
# day_zero + np.timedelta64(273,'D') # 1 May 2015
# day_zero + np.timedelta64(457,'D') # 1 Nov 2015

day_zero = np.datetime64('2014-08-01')
train['date'] = days
train['date'] = pd.to_datetime(train['date'].apply(lambda x: day_zero + np.timedelta64(x,'D')))
# train['month'] = train['date'].apply(lambda x: x.month)
train['dayofweek'] = train['date'].apply(lambda x: x.dayofweek)

time = train.tr_datetime.apply(lambda x: x[-8:]).astype(str)
midnight_indicator = time == '00:00:00'

term_id_is_null_indicator = train['term_id'].isnull().values

dummy_table = pd.get_dummies(train['dayofweek'])
dummy_table['midnight_indicator'] = midnight_indicator
dummy_table['term_id_is_null_indicator'] = term_id_is_null_indicator * 1
dummy_table['customer_id'] = train['customer_id']
dummy_count = dummy_table.groupby('customer_id').agg(np.sum).values

sum_pos = all_pos.sum(axis=1)
sum_neg = all_neg.sum(axis=1)
sub_abs = sum_pos - sum_neg


# term_id_is_null_sum_pos =

plt.plot(np.log(train['term_id'].value_counts().values[:]))

train['term_id'].value_counts().index.values[:300]

train['term_id'].value_counts()[:60]
train['term_id'].value_counts()[60:120]
train['term_id'].value_counts()[120:180]
train['term_id'].value_counts().values#[:30]

sns.distplot(train['term_id'].value_counts().values)

train.loc[term_id_is_null,'mcc_code'].value_counts()[:20]
train.loc[~term_id_is_null,'mcc_code'].value_counts()[:20]





# </editor-fold>

# </editor-fold>

# <editor-fold desc="Saving">
joblib.dump(all_count,'Data/all_count')
joblib.dump(all_pos,'Data/all_pos')
joblib.dump(all_neg,'Data/all_neg')
joblib.dump(all_mean_pos,'Data/all_mean_pos')
joblib.dump(all_mean_neg,'Data/all_mean_neg')
joblib.dump(ids,'Data/ids')
# joblib.dump(all_count_new,'Data/all_count_new')
# joblib.dump(all_addition,'Data/all_addition')
# joblib.dump(target,'Data/target')
# joblib.dump(train_mask,'Data/train_mask')
# </editor-fold>

# <editor-fold desc="Loading">
all_count = joblib.load('Data/all_count')
all_pos = joblib.load('Data/all_pos')
all_neg = joblib.load('Data/all_neg')
all_mean_pos = joblib.load('Data/all_mean_pos')
all_mean_neg = joblib.load('Data/all_mean_neg')
ids = joblib.load('Data/ids')
# all_count_new = joblib.load('Data/all_count_new')
# all_addition = joblib.load('Data/all_addition')
target = joblib.load('Data/target')
train_mask = joblib.load('Data/train_mask')
# </editor-fold>

train_ids = ids[train_mask]; test_ids = ids[~train_mask]
all = np.hstack([all_count,all_pos,all_neg,all_mean_pos,all_mean_neg])
x_train_1 = all[train_mask]; x_test = all[~train_mask]
sc = StandardScaler()
x_norm_train = sc.fit_transform(x_train_1); x_norm_test = sc.fit_transform(x_test)

# <editor-fold desc="Distances">
def pick_smallest(data, list = None, greatest = False):
    '''
    Finding smallest elements in each row of data according to list
    Complexity doesn't depend on len of list
    data : 2-dim np.array
    list : list, np.array - list of minimum orders for output
    '''

    indices = np.argsort(data, axis=1)
    shape1 = list.__len__()
    result = np.array([[0.0] * shape1] * data.shape[0])
    for line in range(data.shape[0]):
        for num in range(shape1):
            if (greatest == False): place = list[num]
            else: place = data.shape[1] + 1 - list[num]
            result[line,num] = data[line,indices[line][place]]
    return result

def find_n_neighbors_multiclass(test,train,target_class,list_neighbors,verbosity=True,substractions=False):
    '''
    finding distances from each element of test to n-th further element of train part with corresponding target_class
    complexity don't depends on len of list_neighbors
    test, train: 2-dim np.arrays
    target_class: corresponding classes of train
    list_neighbors: list or np.array
    verbosity : whether to print finish message
    substractions : whether to include substractions/relatives of distances to different classes. Works ONLY for n_classes = 2
    '''

    result = np.array([[0] * 0]* test.shape[0])
    for n_class in range(n_classes):
        part = train[target_class == n_class]
        distances = pairwise_distances(test,part,metric='euclidean',n_jobs=-1)
        result = np.hstack([result,pick_smallest(distances, list = list_neighbors)])

    if substractions:
        addition_1 = result[:,:list_neighbors.__len__()] - result[:,list_neighbors.__len__():]
        addition_2 = result[:, :list_neighbors.__len__()] / result[:, list_neighbors.__len__():]
        result = np.hstack([result, addition_1, addition_2])

    if verbosity: print('Neighbors found')
    return result

def find_n_neighbors_multiclass_meta(train,target,list_neighbors,n_folds=2,verbosity=True,substractions=False):
    # It is a meta-predictor of the find_n_neighbors_multiclass function. O(1-1/n) complexity
    # substractions : whether to include substractions/relatives of distances to different classes. Works ONLY for n_classes = 2
    if substractions: lenght = list_neighbors.__len__() * 4
    else: lenght = n_classes  * list_neighbors.__len__()

    # Generating indices
    # split_list = list(KFold(n_splits=n_folds, shuffle=True, random_state=0))
    split_list = list(KFold(n=train.shape[0], n_folds=n_folds, shuffle=True, random_state=0))
    result = np.array([[0.0] * (lenght)] * train.shape[0])
    for num in range(n_folds):
        # Spliting train for train, validation and test folds
        train_fold_indices = split_list[num][0]
        test_fold_indices = split_list[num][1]
        train_fold = train[train_fold_indices]
        test_fold = train[test_fold_indices]
        train_target_fold = target[train_fold_indices]
        # test_target_fold = target.iloc[test_fold_indices]
        test_fold_distances = find_n_neighbors_multiclass(test_fold,train_fold,train_target_fold,list_neighbors,verbosity=False,substractions=substractions)
        result[test_fold_indices] = test_fold_distances
        if verbosity: print("Step №", num, "done")
    print("Cross_val_distance done!")
    return result

list_neighbors = 2**np.arange(10) - 1

distances_test = find_n_neighbors_multiclass(x_norm_test,x_norm_train,target,list_neighbors,substractions=True)
distances_train = find_n_neighbors_multiclass_meta(x_norm_train,target,list_neighbors,n_folds=5,substractions=True)

joblib.dump(distances_train,'Data/distances_train')
joblib.dump(distances_test,'Data/distances_test')

distances_train = joblib.load('Data/distances_train')
distances_test = joblib.load('Data/distances_test')
# </editor-fold>

# <editor-fold desc="Meta predictions">
def meta_predict(estimator, X, y, n = 2, verbosity=False):
    '''
    Returns meta-predictions for X with target value y divided by n parts
    O(n-1) or O(1-1/n) or O(len(y))
    estimator - some sklearn estimator or may be smth else with .fit and .predict_proba methods
    X, y - np.arrays
    '''
    split_list = list(KFold(n = X.shape[0], n_folds=n, shuffle=True, random_state=0))
    result = np.array([[0.0] * n_classes] * X.shape[0])
    for num in range(n):
        train_fold_indices = split_list[num][0]
        test_fold_indices = split_list[num][1]
        train_fold = X[train_fold_indices]
        test_fold = X[test_fold_indices]
        train_target_fold = y[train_fold_indices]
        # test_target_fold = y.iloc[test_fold_indices]
        estimator.fit(train_fold,train_target_fold)
        fold_predict = estimator.predict_proba(test_fold)
        #log_loss(test_target_fold, fold_predict)
        result[test_fold_indices] = fold_predict
        if verbosity: print("Step №", num, "done")
    print("Cross_val_predict_proba done!")
    return result

# <editor-fold desc="LR">
lr = LogisticRegression(n_jobs=-1)
lr_predict_train = meta_predict(lr,x_norm_train,target,n=5,verbosity=True)[:,1].reshape(-1,1)
lr.fit(x_norm_train,target)
lr_predict_test = lr.predict_proba(x_norm_test)[:,1].reshape(-1,1)

print(roc_auc_score(target,lr_predict_train))

joblib.dump(lr_predict_train, 'Data/lr_predict_train')
joblib.dump(lr_predict_test, 'Data/lr_predict_test')
# </editor-fold>

# <editor-fold desc="KNN">
knn_predict_test = np.zeros((x_norm_test.shape[0],0))
knn_predict_train = np.zeros((x_norm_train.shape[0], 0))

for n_neighbors in 3**np.arange(1,6):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights='distance',n_jobs=-1)
    knn_predict_train_addition = meta_predict(knn,x_norm_train,target,n=5,verbosity=True)[:,1].reshape(-1,1)
    print(roc_auc_score(target,knn_predict_train_addition))
    knn_predict_train = np.hstack([knn_predict_train,knn_predict_train_addition])
    knn.fit(x_norm_train, target)
    knn_predict_test_addition = knn.predict_proba(x_norm_test)[:,1].reshape(-1,1)
    knn_predict_test = np.hstack([knn_predict_test,knn_predict_test_addition])

joblib.dump(knn_predict_train, 'Data/knn_predict_train')
joblib.dump(knn_predict_test, 'Data/knn_predict_test')
# </editor-fold>

# </editor-fold>

lr_predict_train = joblib.load('Data/lr_predict_train').reshape(-1,1)
lr_predict_test = joblib.load('Data/lr_predict_test').reshape(-1,1)
knn_predict_train = joblib.load('Data/knn_predict_train')
knn_predict_test = joblib.load('Data/knn_predict_test')


x_train_new = np.hstack([x_train_1,lr_predict_train])
x_test_new = np.hstack([x_test,lr_predict_test])

dtrain = xgb.DMatrix(x_train_new,target)
dtest = xgb.DMatrix(x_test_new)

num_trees = 5000; n_fold = 4
for depth in np.arange(1,6):
    print('depth = '+str(depth))#+' ss = '+str(ss)+' cs = '+str(cs))
    params = {"booster": "gbtree",
              "objective": "binary:logistic",
              "eta": 0.3,
              "max_depth": depth,
              "subsample": 0.95,
              "colsample_bytree": 0.95,
              "silent": 1,
              "seed": 1,
              "eval_metric": 'auc'
              }
    print(xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-mean'].values[-1],
          xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-std'].values[-1])


rstate = 0
x_train, x_holdout, y_train, y_holdout = train_test_split(x_train_1, target, test_size=0.2, random_state=rstate)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=rstate)

rf = RandomForestClassifier(n_estimators=600,max_depth=20,n_jobs=-1)
rf.fit(x_train,y_train)

rf_predict = rf.predict_proba(x_val)[:,1]
roc_auc_score(y_val,rf_predict)


dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dholdout = xgb.DMatrix(x_holdout, y_holdout)
dtest = xgb.DMatrix(x_test)
watchlist = [(dval,'eval')]

params = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 2,
          "subsample": 0.95,
          "colsample_bytree": 0.95,
          "silent": 1,
          "seed": 1,
          "eval_metric": 'auc'
          }

gbm = xgb.train(params, dtrain, num_trees, evals = watchlist, early_stopping_rounds = 15, verbose_eval = True)

importance = gbm.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
featp = df.iloc[-50:,].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

top_n_features = df.loc[:,'feature'].apply(lambda x: int(x[1:])).values[-120:]

x_top_train = x_train_1[:,top_n_features]
x_test = all[~train_mask][:,top_n_features]

holdout_predict = gbm.predict(dholdout, ntree_limit = gbm.best_ntree_limit)
val_predict = gbm.predict(dval, ntree_limit = gbm.best_ntree_limit)

print(roc_auc_score(y_val, val_predict))
print(roc_auc_score(y_holdout, holdout_predict))

rstate = 2
x_train, x_holdout, y_train, y_holdout = train_test_split(x_top_train, target, test_size=0.2, random_state=rstate)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=rstate)

dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dholdout = xgb.DMatrix(x_holdout, y_holdout)
dtest = xgb.DMatrix(x_test)
watchlist = [(dval,'eval')]

num_trees = 5000; n_fold = 4
for depth in np.arange(1,6):
    print('depth = '+str(depth))#+' ss = '+str(ss)+' cs = '+str(cs))
    params = {"booster": "gbtree",
              "objective": "binary:logistic",
              "eta": 0.3,
              "max_depth": depth,
              "subsample": 0.95,
              "colsample_bytree": 0.95,
              "silent": 1,
              "seed": 1,
              "eval_metric": 'auc'
              }
    print(xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-mean'].values[-1],
          xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-std'].values[-1])









test_predict = gbm.predict(dtest)










sub = pd.DataFrame(data = test_ids, columns=['customer_id'])
sub['gender'] = np.zeros(test_ids.shape[0],dtype=int)
sub.to_csv('sub_A/sub_only_zeros.csv', index=False)






















train_new = all[train_mask]
x_test = all[~train_mask]

rstate = 1
x_train, x_holdout, y_train, y_holdout = train_test_split(train_new, target, test_size=0.2, random_state=rstate)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=rstate)

dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dholdout = xgb.DMatrix(x_holdout, y_holdout)
dtest = xgb.DMatrix(x_test)
watchlist = [(dval,'eval')]

num_trees = 5000; n_fold = 8
for depth in np.arange(2,4):
    print('depth = '+str(depth))#+' ss = '+str(ss)+' cs = '+str(cs))
    params = {"booster": "gbtree",
              "objective": "binary:logistic",
              "eta": 0.3,
              "max_depth": depth,
              "subsample": 0.95,
              "colsample_bytree": 0.95,
              "silent": 1,
              "seed": 1,
              "eval_metric": 'auc'
              }
    print(xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-mean'].values[-1],xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-std'].values[-1])

params = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eta": 0.02,
          "max_depth": 2,
          "subsample": 0.95,
          "colsample_bytree": 0.95,
          "silent": 1,
          "seed": 1,
          "eval_metric": 'auc'
          }

gbm = xgb.train(params, dtrain, num_trees, evals = watchlist, early_stopping_rounds = 15, verbose_eval = False)

holdout_predict = gbm.predict(dholdout)
val_predict = gbm.predict(dval)

print(roc_auc_score(y_val, val_predict))
print(roc_auc_score(y_holdout, holdout_predict))

test_predict = gbm.predict(dtest)

sub = pd.DataFrame(data = test_ids, columns=['customer_id'])
sub['gender'] = test_predict
sub.to_csv('sub_A/sub_zero.csv', index=False)

importance = gbm.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
featp = df.iloc[-50:,].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))

top_n_features = df.loc[:,'feature'].apply(lambda x: int(x[1:])).values[-150:]
plt.plot(df['fscore'])

top_train = train_new[:,top_n_features]
x_test = all[~train_mask][:,top_n_features]

x_train, x_holdout, y_train, y_holdout = train_test_split(top_train, target, test_size=0.2, random_state=rstate)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=rstate)

dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dholdout = xgb.DMatrix(x_holdout, y_holdout)
dtest = xgb.DMatrix(x_test)
watchlist = [(dval,'eval')]

num_trees = 5000; n_fold = 8
for depth in np.arange(2,4):
    print('depth = '+str(depth))#+' ss = '+str(ss)+' cs = '+str(cs))
    params = {"booster": "gbtree",
              "objective": "binary:logistic",
              "eta": 0.3,
              "max_depth": depth,
              "subsample": 0.95,
              "colsample_bytree": 0.95,
              "silent": 1,
              "seed": 1,
              "eval_metric": 'auc'
              }
    print(xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-mean'].values[-1],xgb.cv(params,dtrain,num_trees,n_fold,early_stopping_rounds = 6)['test-auc-std'].values[-1])

test_predict = gbm.predict(dtest)




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

