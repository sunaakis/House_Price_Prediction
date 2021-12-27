#####################################################################
# UNSUPERVISED MACHINE LEARNING - HOUSE PRICE PREDICTION
#####################################################################
# If the problem is a classification problem, scaling the independent variable has no meaning or importance.
# But if it is a regression problem, not cart, but light gbm, that is, if there is an optimization problem;
# The size of the variables may be important for ordering, it can be faster to optimize when you standardize
# the dependent variable

######################################
# Libraries
######################################
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

######################################
# Assembling the train and test sets.
######################################

train = pd.read_csv("Datasets/house_price_train.csv")
test = pd.read_csv("Datasets/house_price_test.csv")
test.head()

train.head()
df = train.append(test).reset_index(drop=True)
df.head()
# This normally causes data leakage.
# We brought the two datasets together and did the engineering on the new dataset
# The theoretical problem of processing train and test together is that you have trained
# the set that you would normally only train with the train with more data and the data from the test,
# and the information from the test set has leaked into the train set, this is called data leakage.
# You have normalized, but when you separate it back, this set is now correlated between two data due to data leakage


######################################
# EDA
######################################
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# see quantile values : year variables, MasVnrArea, BsmtFinSF1, TotalBsmtSF, MiscVal, LotArea,
# # variables we need to modify

# In general, it may be necessary to reduce only the most outliers from the end,
# those that correspond to values of 0.01 and 0.99.

def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = # of variables.
    # num_but_cat is already in cat_cols.
    # we can select all variables by only the following 3 lists: cat_cols + num_cols + cat_but_car
    # num_but_cat is only for reporting.

    return cat_cols, cat_but_car, num_cols, num_but_cat


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
# Observations: 2919
# Variables: 81
# cat_cols: 50
# num_cols: 28
# cat_but_car: 3
# num_but_cat: 10

# it will be necessary to convert the year variables,
# You need to leave out the id variable because it's not num_col

######################################
# Categorical Variable Analysis
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# they may be meaningless:
# GarageCars_5
# Fireplaces_4


for col in cat_but_car:
    cat_summary(df, col)

# Exterior1st-2nd, for example, it has many classes, it may not be meaningful or we can classify it if it is meaningful
# most significant variable neighborhood

for col in num_but_cat:
    cat_summary(df, col)


######################################
# Numerical Variable Analysis
######################################

df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99]).T
# LotArea, BsmtFinSF2, LowQualFinSF, 3SsnPorch, ScreenPorch, MiscVal


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)

######################################
# Target Variable Analysis
######################################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])


def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

high_corrs
# ['OverallQual: 0.7909816005838047',
#  'TotalBsmtSF: 0.6135805515591944',
#  '1stFlrSF: 0.6058521846919166',
#  'GrLivArea: 0.7086244776126511',
#  'GarageArea: 0.6234314389183598']

low_corrs
# ['Id: -0.021916719443431112',
#  'MSSubClass: -0.08428413512659523',
#  'LotFrontage: 0.35179909657067854',
#  'LotArea: 0.2638433538714063',....

# variables that can be thrown away:
# MSSubClass, BsmtFinSF2, LowQualFinSF, EnclosedPorch, 3SsnPorch, PoolArea, MiscVal, MoSold


######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################

len(df.columns)
df.shape
# import itertools
# combn = list(itertools.combinations(df.columns, 5))


# Basement floor area + Above ground area
df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

# Change Format
# df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y')
# df['YearRemodAdd'] = pd.to_datetime(df['YearRemodAdd'], format='%Y')
# df['GarageYrBlt'] = pd.to_datetime(df['GarageYrBlt'], format='%Y')
# df['YrSold'] = pd.to_datetime(df['YrSold'], format='%Y')

# df['GarageYrBlt'] = np.where(df['GarageYrBlt'].isnull(), df['YearBuilt'], df['GarageYrBlt'])
# If the garage construction year is blank, put the building construction year.

# 1 if there is a garage, 0 if not
df["Grg_Existence"] = np.where(df['GarageYrBlt'].isnull(), 0, 1)

##################
# Building Age
##################

df['YrSold'].max() # Date of sale
# current_date = pd.to_datetime('2010-01-01 0:0:0')
df['BuiltAge'] = (df['YrSold'].max() - df['YearBuilt'])
df['Sold-Built'] = (df['YrSold'] - df['YearBuilt'])
df['Sold-RemodAdd'] = (df['YrSold'] - df['YearRemodAdd'])

# Was the building renovated when it was sold?
# it can be a variable
# get building age as the day it was renovated - create new variable

df['Sold-RemodeAdd'].hist(bins=50)
plt.show()
# That is, after the building was sold, it was renovated-
# The building's age is when it was renovated

##################
# Building Type
##################

df["BldgType"].value_counts()
# 1Fam      2425 Single-family Detached
# TwnhsE     227 Townhouse End Unit
# Duplex     109
# Twnhs       96 Townhouse Inside Unit
# 2fmCon      62 Two-family Conversion
df["TownHouse"] = np.where((df['BldgType'] != "TwnhsE") & (df['BldgType'] != "Twnhs"), 0, 1)

df["Bsmt"] = np.where((df['BsmtCond'].isnull()), 0, 1)

##################
# Basement availability
##################

df.loc[(df['BsmtFinType1'].isnull()),"Bsmt_Usability" ] = 0
df.loc[(df['BsmtFinType1'] == "LwQ"), "Bsmt_Usability" ] = 0
df.loc[(df['BsmtFinType1'] == "Unf"), "Bsmt_Usability" ] = 0
df["Bsmt_Usability"] = np.where((df['Bsmt_Usability'] == 0), 0, 1)
df["Bsmt_Usability"].value_counts()
# 1    1835
# 0    1084

##################
# Heating quality
##################

df["HeatingQC"].value_counts()

clean = {"Ex":"above_avg", "Gd":"above_avg",
        "TA": "avg", "Fa": "below_avg", "Po": "below_avg"}


df["Heat_Cat"] = df["HeatingQC"].replace(clean)
df["Heat_Cat"].value_counts()
# above_avg    1967
# avg           857
# below_avg      95

##################
# Bedroom & Bathroom
##################

df["Bedroom_Num"].value_counts()

df.loc[df["BedroomAbvGr"] >= 4, "Bedroom_Num"] = "luxury"
df.loc[(df["BedroomAbvGr"] == 3), "Bedroom_Num"] = "Normal"
df.loc[df["BedroomAbvGr"] <= 2 , "Bedroom_Num"] = "Small"


df["Room_Ratio"] = df["BedroomAbvGr"] / df["GrLivArea"] *100
df["Room_Ratio"].describe()


df["TotalFullBath"] = df["BsmtFullBath"] + df["FullBath"]
df["TotalHalfBath"] = df["BsmtHalfBath"] + df["HalfBath"]

##################
# Season effect on sales date
##################

df["MoSold_Seasons"] = pd.cut(df["MoSold"], bins =[0,3,6,10,12], labels=["winter","spring","summer","autumn"])
df["MoSold_Seasons"].value_counts()
# df["MoSold"] The month variable is practically not a continuous variable.
# It is necessary to act categorically: June is not twice as much as March.

df.groupby("MoSold").agg({"SalePrice":["count","mean","median"]})
# 9-10-11 can be a category, etc.

##################
# miscellaneous
##################
# We can also do 1-0 for the Fence variable instead of removing it completely from the data.

misc_list = ["PoolQC", "MiscFeature", "Alley","Fence", "FireplaceQu", "LotFrontage"]
df["Fireplace"] = np.where(df['FireplaceQu'].isnull(), 0, 1)
df["Pool"] = np.where(df['PoolQC'].isnull(), 0, 1)
df["Fence_VarMi"] = np.where(df['Fence'].isnull(), 0, 1)


drop_list= ["MoSold","TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath","BedroomAbvGr",
            "GrLivArea", "HeatingQC", "BsmtFinType1", "BldgType","BsmtCond","MiscFeature","Fence",
            "YrSold","YearRemodAdd","YearBuilt", "GarageYrBlt", "FireplaceQu","PoolQC" ,"Alley",
            "1stFlrSF","2ndFlrSF", "BsmtFinSF1","BsmtFinSF2", "BsmtUnfSF", "RoofStyle", "RoofMatl"]

for i in drop_list:
    df.drop(i, axis=1, inplace=True)



sns.heatmap(corr_data)
plt.show()

##################
# categorize the variables again
##################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)

##################
# Missing Values
##################

na_list = pd.DataFrame()
for col in df.columns:
    if df[col].isnull().any() == True:
        na_sum = df[col].isnull().sum()
        new_tb=pd.DataFrame({"na_sum": [na_sum]}, index=[col])
        na_list = pd.concat([na_list, new_tb])


cat_na = [col for col in list(na_list.index) if col in cat_cols]
num_na = [col for col in list(na_list.index) if col in num_cols and "SalePrice" not in col]

na_list.sort_values("na_sum", ascending=False)

# drop_list = ["PoolQC", "MiscFeature", "Alley","Fence", "FireplaceQu", "LotFrontage"]


df[num_na] = df[num_na].apply(lambda x: x.fillna(x.median()), axis=0)
df[cat_na] = df[cat_na].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

df.isnull().sum()
df.head()

######################################
# RARE ENCODING
######################################

# For use for the scarcity of categories within classes:

def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")



rare_analyser(df, "SalePrice", 0.01)
# if the ratio of any of these classes to the whole data is less than 0.01:
# MSZoning : 5
#          COUNT  RATIO  TARGET_MEAN
# C (all)     25  0.009    74528.000
# FV         139  0.048   214014.062
# RH          26  0.009   131558.375
# RL        2265  0.776   191004.995
# RM         460  0.158   126316.830
# ....

# It is necessary to construct the model in the simplest way possible, for example,
# C (all) and RH can be combined because their proportions are very small.

# Use itertools for the Neighborhood variable and have the target mean cleaned accordingly

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.01)

df["Neighborhood"].value_counts()
df.groupby("Neighborhood").agg({"SalePrice":"mean"}).sort_values("SalePrice", ascending=False)

######################################
# LABEL ENCODING & ONE-HOT ENCODING & SCALING
######################################

# normally these were separate now we need to add them because we have completed the process
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)
cat_cols = cat_cols + cat_but_car


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

######################################
# MISSING_VALUES
######################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

# If the dataset is large, touching the missing data breaks the structure, if the dataset is small, we can fill it

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

# it may be dependant structure but we filled in the median for practicality



######################################
# OUTLIERS
######################################


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

outlier_list = [col for col in num_cols if check_outlier(df, col) == True ]


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


for col in outlier_list:
    replace_with_thresholds(df, col)

df.head()

######################################
# Variable Comprehension
######################################
abc = df.corr()["SalePrice"].sort_values(ascending = False)


######################################
# Quick Try
######################################
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)
import statsmodels.api as sm

y = np.log1p(train_df["SalePrice"])
X = train_df.drop(["Id", "SalePrice"], axis=1)
model = sm.OLS(y, X)
model_fit = model.fit()
p_values = model_fit.summary2().tables[1]["P>|t|"]
p_values.sort_values( ascending=True)

error_table = pd.DataFrame()
for col in train_df.columns:
    y = np.log1p(train_df["SalePrice"])
    X = train_df[[col]]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # lin_model = LinearRegression().fit(X_train, y_train)
    model = sm.OLS(y, X)
    model_fit = model.fit()
    p_values = model_fit.summary2().tables[1]["P>|t|"]
    # y_pred = lin_model.predict(X_test)
    # error = np.sqrt(mean_squared_error(y_test, y_pred))
    # coef = lin_model.coef_
    new_table = pd.DataFrame({"p_values": [p_values]}, index=[col])
    error_table = pd.concat([error_table, new_table])
print(error_table.sort_values("p_values", ascending=False))

ftr = error_table.loc[(error_table["importance"] >= 0.10), :]
liste = list(ftr.index)

from lightgbm import LGBMRegressor
y = np.log1p(train_df['SalePrice'])
X = train_df[liste]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)


lgb_model = LGBMRegressor(random_state=42).fit(X_train, y_train)
y_pred = lgb_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#  0.11178253975422528
y_pred = lgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 0.15699160898621756

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}

lgb_model = LGBMRegressor(random_state=42)
lgb_tuned = GridSearchCV(lgb_model, lgbm_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
lgb_tuned.best_params_
# {'colsample_bytree': 0.5,
#  'learning_rate': 0.01,
#  'max_depth': 8,
#  'n_estimators': 1000}

lgb_tuned = LGBMRegressor(**lgb_tuned.best_params_).fit(X_train, y_train)

y_pred = lgb_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 0.1293009671442434
y_pred = lgb_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 0.14930857405584774



######################################
# TRAIN - TEST SPLIT
######################################

train_df = df[df["SalePrice"].notnull()]
test_df = df[df["SalePrice"].isnull()].drop("SalePrice", axis=1)

# Let's save it as prepared data so we don't lose it
train_df.to_pickle("Datasets/house_train_df.pkl")
test_df.to_pickle("Datasets/house_test_df.pkl")

#######################################
# MODEL: LGBM
#######################################
from lightgbm import LGBMRegressor

X = train_df.drop(['SalePrice', "Id"], axis=1)
#y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=46)

lgbm_model = LGBMRegressor(random_state=42).fit(X_train, y_train)
y_pred = lgbm_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#   0.04326144864623516

# Error calculation with test set:
y_pred = lgbm_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 0.12512462660361628

# almost doubled output, we always expect the test error to come out worse
# but here this error is not very good 10 thousand was reasonable but 27 thousand high so model needs improvement

# we did not do model hyperparameter optimization
# data leakage problem, all of these may have increased this error

#######################################
# Model Tuning
#######################################

lgbm_model = LGBMRegressor()
lgbm_params = {"learning_rate": [0.01, 0.1], # Try 0.2 and try to catch the middle one -0.2
               "n_estimators": [300, 500], # estimator may be under 500 look- 300
               "max_depth": [3, 5, 8], # It can be less than 3
               "colsample_bytree": [1, 0.8, 0.5]}

lgbm_tuned = GridSearchCV(lgbm_model,lgbm_params,cv=10, n_jobs=-1,
                          verbose=2).fit(X_train, y_train)

lgbm_tuned.best_params_
# {'colsample_bytree': 0.5,
#  'learning_rate': 0.1,
#  'max_depth': 3,
#  'n_estimators': 500}

lgbm_tuned = LGBMRegressor(**lgbm_tuned.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 0.11344181862557261

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_tuned.best_params_).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 0.07347391437362907
y_pred = lgbm_tuned.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 0.11344181862557261

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_tuned, X_train, 20)










