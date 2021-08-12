## PROBLEM DEFINITION: predict sale price of bulldozer using ML and previous data
## DATA: kaggle blue-book for bulldozers competition
## EVALUATION: minimize( root mean square log error between actual and predicted value0
## FEATURES: Kaggle data dictionary

## COLLECTING TOOLS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## EXPLORATORY DATA ANALYSIS

# df= pd.read_csv('TrainAndValid.csv',
#                 low_memory=False)
# print(df.info)
# print(df.columns)
# fig, ax= plt.subplots()
# ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000])
# df.SalePrice.plot.hist()
# plt.show()

df= pd.read_csv('TrainAndValid.csv',
                low_memory=False,
                parse_dates= ['saledate'])
# fig, ax= plt.subplots()
# ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000])
# plt.show()

# df.sort_values(by= ['saledate'], inplace= True, ascending=True)
df_tmp= df.copy()
# print(df.columns)
df_tmp['SaleYear'] = df_tmp.saledate.dt.year
df_tmp['SaleMonth'] = df_tmp.saledate.dt.month
df_tmp['SaleDay'] = df_tmp.saledate.dt.day
df_tmp['SaleDayoftheWeek'] = df_tmp.saledate.dt.dayofweek
df_tmp['SaleDayofYear'] = df_tmp.saledate.dt.dayofyear
df_tmp.drop('saledate',axis=1,inplace=True)

# print(df_tmp.columns)
# print(df_tmp.state.value_counts())

# TURNING DATA TO NUMERIC

# print(pd.api.types.is_string_dtype(df_tmp['UsageBand']))

# for label,content in df_tmp.items():
#     if pd.api.types.is_string_dtype(content):
#         print(label)

for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()

# print(df_tmp.info)
# print(df_tmp.state.cat.categories)
# print(df_tmp.state.cat.codes)
# print(df_tmp.isnull().sum())

## SAVING DATA
df_tmp.to_csv('Train_tmp.csv', index= False)

# RETRIEVE PROCESSED DATA
df_tmp= pd.read_csv('train_tmp.csv', low_memory= False)

# FILL NULL DATA ENTRIES THAT ARE OF NUMERIC TYPE
for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label + '_is_missing'] = pd.isnull(content)
            df_tmp[label] = content.fillna(content.median())

# FILL NULL DATA ENTRIES THAT ARE NOT IN NUMERIC TYPE DATA COLUMN
# for label, content in df_tmp.items():
#     if not pd.api.types.is_numeric_dtype(content):
#         print(label)

for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+'_is_missing'] = pd.isnull(content)
        df_tmp[label] = pd.Categorical(content).codes+1

# print(df_tmp.info)
# print(df_tmp.isna().sum())

## MODELLING

# import time # to show how much time it takes to go through 412000 rows
from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_jobs=-1, random_state= 31)
# start_time = time.time()
# model.fit(df_tmp.drop('SalePrice',axis= 1), df_tmp['SalePrice'])
# print(model.score(df_tmp.drop('SalePrice',axis= 1), df_tmp['SalePrice']))
# end_time = time.time()
# print(f'elapsed time : {(end_time - start_time):.2f} ms')

## SPLITTING DATA INTO TRAINING AND VALIDATION (2012 DATA)
df_val = df_tmp[df_tmp['SaleYear'] == 2012]
df_train = df_tmp[df_tmp['SaleYear'] != 2012]
x_val, y_val = df_val.drop('SalePrice', axis= 1), df_val['SalePrice']
x_train, y_train = df_train.drop('SalePrice', axis= 1), df_train['SalePrice']

## EVALUATION
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test,y_preds))

def show_scores(model):
    train_preds = model.predict(x_train)
    val_preds = model.predict(x_val)
    scores = {'Training MAE': mean_absolute_error(y_train,train_preds),
              'Valid MAE': mean_absolute_error(y_val,val_preds),
              'Training RMSLE': rmsle(y_train, train_preds),
              'Valid RMSLE': rmsle(y_val,val_preds),
              'Training R*2': r2_score(y_train,train_preds),
              'Valid R*2': r2_score(y_val,val_preds)}
    return scores

# model= RandomForestRegressor(n_jobs=-1, random_state=31, max_samples=100)
# model.fit(x_train,y_train)
# print(show_scores(model))

## RANDOMIZED SEARCH CV
from sklearn.model_selection import RandomizedSearchCV
# rf_grid= {'n_estimators': np.arange(10,100,10),
#           'max_depth': [None,3,5,10],
#           'min_samples_split': np.arange(2,20,2),
#           'min_samples_leaf': np.arange(1,20,2),
#           'max_features': [0.5,1,'sqrt','auto'],
#           'max_samples': [None,100]}
# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=31),
#                               param_distributions=rf_grid,
#                               n_iter=20,
#                               cv=5,
#                               verbose= True)
# rs_model.fit(x_train,y_train)
# print(show_scores(rs_model))
# print(rs_model.best_params_)

## TEST DATA PREPROCESSING
ideal_model = RandomForestRegressor(n_estimators=20,
                                    min_samples_split=10,
                                    min_samples_leaf=5,
                                    max_samples=100,
                                    max_features=0.5,
                                    max_depth=10,
                                    n_jobs=-1)
ideal_model.fit(x_train,y_train)
df_test = pd.read_csv('Test.csv',low_memory=False,parse_dates=['saledate'])
# test_preds = ideal_model.predict(df_test)

def preprocess_data(df):
    df['SaleYear'] = df.saledate.dt.year
    df['SaleMonth'] = df.saledate.dt.month
    df['SaleDay'] = df.saledate.dt.day
    df['SaleDayoftheWeek'] = df.saledate.dt.dayofweek
    df['SaleDayofYear'] = df.saledate.dt.dayofyear
    df.drop('saledate',axis= 1, inplace=True)
    for label,content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label + '_is_missing'] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        if not pd.api.types.is_numeric_dtype(content):
            df[label + '_is_missing'] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes+1
    return df

df_test = preprocess_data(df_test)
# print(set(x_train.columns)-set(df_test.columns))
df_test['auctioneerID_is_missing'] = False

## MAKING PREDICTIONS USING TEST DATA
test_preds = ideal_model.predict(df_test)

## SETTING TEST DATA TO THE FORMAT
df_preds = pd.DataFrame()
df_preds['SalesID'] = df_test['SalesID']
df_preds['SalesPrice'] = test_preds
# print(df_preds)

## EXPORTING CSV
df_preds.to_csv('test_predictions.csv', index= False)

## FEATURE IMPORTANCE
# print(ideal_model.feature_importances_)

import seaborn as sns
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({'features': columns,
                       'feature_importance': importances})
          .sort_values('feature_importance', ascending=False)
          .reset_index(drop=True))
    fig, ax = plt.subplots()
    ax.barh(df['features'][:n], df['feature_importance'][:n])
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature_Importance')
    ax.invert_yaxis()
    plt.show()

plot_features(x_train.columns, ideal_model.feature_importances_)
# print(sum(ideal_model.feature_importances_))
# print(df.ProductSize.value_counts())
