# Functionality
import pandas as pd
import numpy as np

# Provides functions for interacting with the operating system
import os

# For Zillow data null values
import random

# statistical modeling
import scipy.stats as stats

# To acquire MYSQL Data
import acquire
from env import username, password, host

# For data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# For modeling
import sklearn.metrics as mtc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector

# --------------------------------------------------
'''Establishing a overarching SQL server contact method'''

# Function to pull data from SQL
def get_db_url(username, hostname, password, database):
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

# Function used to split data when modeling
def split(df):
    train, test = train_test_split(df, test_size=.2, random_state=248)
    train, validate = train_test_split(train, test_size=.25, random_state=248)
    print(f'{df} shape: {df.shape}')
    print(f'Train shape: {train.shape}')
    print(f'Validate shape: {validate.shape}')
    print(f'Test shape: {test.shape}')
    return train, validate, test

# --------------------------------------------------
'''Zillow data'''

def wrangle_zillow(df):
    random.seed(248)
    df = df[(df.propertylandusedesc == 'Single Family Residential') | 
            (df.propertylandusedesc == 'Inferred Single Family Residential')]
    # Create a new column for month of transaction
    df['month'] = df['transactiondate'].str[5:7]
    df['transactiondate'] = df['transactiondate'].str[:4]
    # Reassign the df to only the values where the home had a transaction in 2017
    df = df[df.transactiondate == '2017']
    df = df[['parcelid','bathroomcnt', 'bedroomcnt','buildingqualitytypeid', 'calculatedfinishedsquarefeet','fips',
             'lotsizesquarefeet', 'regionidzip', 'taxvaluedollarcnt']]
    parcelid = df.parcelid
    df = df.drop(columns={'parcelid'})
    df.rename(columns={'bedroomcnt':'bedrooms',
                       'bathroomcnt':'bathrooms', 
                       'calculatedfinishedsquarefeet':'f_sqft', 
                       'taxvaluedollarcnt':'tax_value',
                       'lotsizesquarefeet':'lot_size',
                       'regionidzip':'zip',
                       'buildingqualitytypeid':'bldg_quality'}, inplace=True)
    cols = [col for col in df.columns if col not in ['month', 'fips']]
    
    df.bedrooms.fillna(random.randint(2.0, 5.0), inplace = True)
    df.bathrooms.fillna(random.randint(1.0, 3.0), inplace = True)
    df.f_sqft.fillna(df.f_sqft.median(), inplace = True)
    df.bldg_quality.fillna(random.randint(6.0, 8.0), inplace=True)
    df.lot_size.fillna(random.randint(5000.0, 8000.0), inplace=True)
    df.zip.fillna(97319.0, inplace=True)
    # Replacing only one value, so I'll input the median
    df.tax_value.fillna(373612.0, inplace=True)
    # Remove outliers
    df = remove_outliers(df, 1.25, df[cols])
    return df

def scale_data(train, 
               validate, 
               test, 
               columns=['bedrooms', 'bathrooms', 'bldg_quality', 'f_sqft', 'zip','lot_size'], return_scaler=False):
    '''
    Scales train, validate, test and returns scaled versions of each 
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # Make the scaler
    scaler = MinMaxScaler()
    # Fit it
    scaler.fit(train[columns])
    # Apply the scaler:
    train_scaled[columns] = pd.DataFrame(scaler.transform(train[columns]),
                                                  columns=train[columns].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns] = pd.DataFrame(scaler.transform(validate[columns]),
                                                  columns=validate[columns].columns.values).set_index([validate.index.values])
    
    test_scaled[columns] = pd.DataFrame(scaler.transform(test[columns]),
                                                 columns=test[columns].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

              
# --------------------------------------------------
# To test columns of a df and see if there is a relationship between the two
def test(df, col, col2):
    alpha = 0.05
    H0 = col+ ' and ' +col2+ ' are independent'
    Ha = 'There is a relationship between ' +col2+ ' and '+col
    observed = pd.crosstab(df[col2], df[col])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject that', H0)
        print(Ha)
    else:
        print('We fail to reject that', H0)
        print('There appears to be no relationship between ' +col2+ ' and '+col)
# --------------------------------------------------
        '''Best Predictors'''

'''Determines the best predictors of your target and returns the column names of the best predictors and a sample dataframe'''
def select_kbest(X_train, y_train, k):
    # create the model
    kbest = SelectKBest(f_regression, k=k)
    # Fit the model
    kbest.fit(X_train, y_train)
    # df of the top predictors
    X_train_transformed = pd.DataFrame(kbest.transform(X_train),
                                       columns=X_train.columns[kbest.get_support()],
                                       index=X_train.index)
    
    return X_train.columns[kbest.get_support()].tolist(), X_train_transformed.head(3)

def rfe(X_train, y_train, k):
    model = LinearRegression()
    # Make the model
    rfe = RFE(model, n_features_to_select=k)
    # Fit the model
    rfe.fit(X_train, y_train)
    # df of the top predictors
    X_train_transformed = pd.DataFrame(rfe.transform(X_train), 
                                       index= X_train.index, 
                                       columns=X_train.columns[rfe.support_])
    
    return X_train.columns[rfe.support_], X_train_transformed.head(3)
        
# --------------------------------------------------
# print("The bold text is",'\033[1m' + 'Python' + '\033[0m')        
# '\033[1m' + 'TEXT' + '\033[0m'
# --------------------------------------------------

# Generic splitting function for continuous target.

def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=248)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=248)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test


def train_validate_test(df, target):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=248)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=248)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = pd.DataFrame(train[target])

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = pd.DataFrame(validate[target])

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = pd.DataFrame(test[target])

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# --------------------------------------------------

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
    print('Percentage of original Dataframe removed: {:.2f}'.format((num_obs - df.shape[0])/52441))
    
    return df

# --------------------------------------------------

def rmse(y_train, y_validate, target):
# 1. Predict value_pred_mean
    value_pred_mean = y_train[target].mean()
    y_train['pred_mean'] = value_pred_mean
    y_validate['pred_mean'] = value_pred_mean

# 2. compute value_pred_median
    pred_median = y_train[target].median()
    y_train['pred_median'] = pred_median
    y_validate['pred_median'] = pred_median

# 3. RMSE of value_pred_mean
    mean_rmse_train = mean_squared_error(y_train[target], y_train.pred_mean)**0.5
    mean_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_mean)**0.5

    # 4. RMSE of value_pred_median
    median_rmse_train = mean_squared_error(y_train[target], y_train.pred_median)**0.5
    median_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_median)**0.5
    
    print('\033[1m' + "\u0332".join('RMSE') + '\033[0m')
    print()
    print('\033[1m' + 'Using Mean:'+ '\033[0m')
    print(f'Train(In-Sample):        {round(mean_rmse_train)}') 
    print(f'Validate(Out-of-Sample): {round(mean_rmse_validate)}')
    print()
    print('\033[1m' + 'Using Median:'+ '\033[0m')
    print(f'Train(In-Sample):        {round(median_rmse_train)}') 
    print(f'Validate(Out-of-Sample): {round(median_rmse_validate)}')
    
# --------------------------------------------------

def pf(X_train, y_train, X_validate, y_validate, X_test, y_test, target, n):
# make the polynomial features to get a new set of features
    n=n
    pf = PolynomialFeatures(degree=n)

# fit and transform X_train_scaled
    X_train_degree = pf.fit_transform(X_train)

# transform X_validate & X_test
    X_validate_degree = pf.transform(X_validate)
    X_test_degree = pf.transform(X_test)

## **LinearRegression**
# create the model object
    lm = LinearRegression(normalize=True)

# fit 
    lm.fit(X_train_degree, y_train[target])

# predict and use train
    y_train['pred_lm'] = lm.predict(X_train_degree)

# evaluate: rmse
    rmse_train = mean_squared_error(y_train[target], y_train.pred_lm) ** (1/2)

# predict validate
    y_validate['pred_lm'] = lm.predict(X_validate_degree)

# evaluate: rmse
    rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_lm) ** (1/2)
    
# Predict Test
    y_test['pred_lm'] = lm.predict(X_test_degree)
    lm_rmse_test = mean_squared_error(y_test[target], y_test.pred_lm) ** (1/2)

    print('\033[1m' + '\u0332'.join('OLS using LinearRegression')+ '\033[0m')
    print()
    print(f'RMSE for Training/In-Sample:        {round(rmse_train)}')
    print(f'RMSE for Validation/Out-of-Sample:  {round(rmse_validate)}')
    print(f'R^2 Validate:                       {round(explained_variance_score(y_validate.tax_value, y_validate.pred_lm), 3)}')
    
# --------------------------------------------------    

def pf_test(X_train, y_train, X_test, y_test, target, n):
    n=n
    pf = PolynomialFeatures(degree=n)

    X_train_degree = pf.fit_transform(X_train)
    X_test_degree = pf.transform(X_test)
    # create the model object
    lm = LinearRegression(normalize=True)

# fit 
    lm.fit(X_train_degree, y_train[target])
    
    
    
    # Predict Test
    y_test['pred_lm'] = lm.predict(X_test_degree)
    lm_rmse_test = mean_squared_error(y_test[target], y_test.pred_lm) ** (1/2)
    print(f'RMSE for Test:  {round(lm_rmse_test)}')
    print(f'R^2 Test:        {round(explained_variance_score(y_test.tax_value, y_test.pred_lm), 3)}')

# --------------------------------------------------

def lasso_lars(X_train, y_train, X_validate, y_validate, X_test, y_test, target, n):
# create the model object
    lars = LassoLars(alpha=n)

# fit the model to our training data.
    lars.fit(X_train, y_train[target])

# predict train
    y_train['pred_lars'] = lars.predict(X_train)

# evaluate: rmse
    lars_rmse_train = mean_squared_error(y_train[target], y_train.pred_lars) ** 0.5

# predict validate
    y_validate['pred_lars'] = lars.predict(X_validate)

# evaluate: rmse
    lars_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_lars) ** 0.5
    
    print('\033[1m' + '\u0332'.join('Lasso + Lars')+ '\033[0m')
    print()
    print(f'RMSE for Training/In-Sample:        {round(lars_rmse_train)}')
    print(f'RMSE for Validation/Out-of-Sample:  {round(lars_rmse_validate)}')
    print(f'R^2 Validate:                       {round(explained_variance_score(y_validate.tax_value, y_validate.pred_lars), 3)}')
# --------------------------------------------------

def glm(X_train, y_train, X_validate, y_validate, X_test, y_test, target, p, a):
# create the model object
    glm = TweedieRegressor(power=p, alpha=a)

# fit the model to our training data. 
    glm.fit(X_train, y_train[target])

# predict train
    y_train['pred_glm'] = glm.predict(X_train)

# evaluate: rmse
    glm_rmse_train = mean_squared_error(y_train[target], y_train.pred_glm) ** 0.5

# predict validate
    y_validate['pred_glm'] = glm.predict(X_validate)

# evaluate: rmse
    glm_rmse_validate = mean_squared_error(y_validate[target], y_validate.pred_glm) ** 0.5
    
    print('\033[1m' + '\u0332'.join('GLM using Tweedie')+ '\033[0m')
    print()
    print(f'RMSE for Training/In-Sample:        {round(glm_rmse_train)}')
    print(f'RMSE for Validation/Out-of-Sample:  {round(glm_rmse_validate)}')
    print(f'R^2 Validate:                       {round(explained_variance_score(y_validate.tax_value, y_validate.pred_glm), 3)}')
# --------------------------------------------------
# --------------------------------------------------

# This function that will be apply to ('bedroomcnt','bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt','yearbuilt','taxamount','fips') on homes that are SFH
def basic_zillow(df):
    df = df[(df.propertylandusedesc == 'Single Family Residential') | (df.propertylandusedesc == 'Inferred Single Family Residential')]
    df = df[['bedroomcnt',
             'bathroomcnt', 
             'calculatedfinishedsquarefeet', 
             'taxvaluedollarcnt']]
    df.rename(columns={'bedroomcnt':'bedrooms',
                       'bathroomcnt':'bathrooms', 
                       'calculatedfinishedsquarefeet':'f_sqft', 
                       'taxvaluedollarcnt':'tax_value'}, inplace=True)
    df.bedrooms.fillna(random.randint(2.0, 5.0), inplace = True)
    df.bathrooms.fillna(random.randint(1.0, 3.0), inplace = True)
    df.f_sqft.fillna(df.f_sqft.median(), inplace = True)
    df.tax_value.fillna(df.tax_value.mode().max(), inplace = True)
    return df


def c_wrangle_zillow(df):
    df = df[(df.propertylandusedesc == 'Single Family Residential') | 
            (df.propertylandusedesc == 'Inferred Single Family Residential')]
    # Create a new column for month of transaction
    df['month'] = df['transactiondate'].str[5:7]
    df['transactiondate'] = df['transactiondate'].str[:4]
    # Reassign the df to only the values where the home had a transaction in 2017
    df = df[df.transactiondate == '2017']
    df = df[['parcelid','bathroomcnt', 'bedroomcnt','buildingqualitytypeid', 'calculatedfinishedsquarefeet','fips',
             'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet', 'regionidzip','yearbuilt',
             'taxvaluedollarcnt', 'month']]
    parcelid = df.parcelid
    df = df.drop(columns={'parcelid'})
    df.rename(columns={'bedroomcnt':'bedrooms',
                       'bathroomcnt':'bathrooms', 
                       'calculatedfinishedsquarefeet':'f_sqft', 
                       'taxvaluedollarcnt':'tax_value',
                       'heatingorsystemtypeid':'systemtype',
                       'lotsizesquarefeet':'lot_size',
                       'regionidzip':'zip',
                       'yearbuilt':'built',
                       'buildingqualitytypeid':'bldg_quality'}, inplace=True)
    cols = [col for col in df.columns if col not in ['month', 'fips']]
    
    df.bedrooms.fillna(random.randint(2.0, 5.0), inplace = True)
    df.bathrooms.fillna(random.randint(1.0, 3.0), inplace = True)
    df.f_sqft.fillna(df.f_sqft.median(), inplace = True)
    # For yearbuilt I'll use 1958 as it falls in the middle of the mean and mode and they are all fairly close in value
    df.bldg_quality.fillna(random.randint(6.0, 8.0), inplace=True)
    df.f_sqft.fillna(1659.0, inplace=True)
    df.systemtype.fillna(2.0, inplace=True)
    df.lot_size.fillna(random.randint(5000.0, 8000.0), inplace=True)
    df.zip.fillna(97319.0, inplace=True)
    # The top 5 most common years are in the 1950's
    df.built.fillna(random.randint(1950.0, 1955.0), inplace=True)
    # Replacing only one value, so I'll input the median
    df.tax_value.fillna(373612.0, inplace=True)
    # Remove outliers
    df = remove_outliers(df, 1.25, df[cols])
    #df.drop(col
    return df

# --------------------------------------------------

def value_viz(df):
    plt.figure(figsize=(16,5))
    plt.hist(df.tax_value[df.fips == 6111.0], color='purple', alpha=.3, label="Ventura", bins=150)
    plt.hist(df.tax_value[df.fips == 6037.0], color='blue', alpha=.3, label="Los Angeles", bins=150)
    plt.hist(df.tax_value[df.fips == 6059.0], color='black', alpha=.3, label="Orange", bins=150)

    plt.axvline(x=df.tax_value[df.fips == 6037.0].mean(), color='blue', label='LA County Mean Value: 356k')
    plt.axvline(x=df.tax_value[df.fips == 6059.0].mean(), color='black', label='Orange County Mean Value: 447k')
    plt.axvline(x=df.tax_value[df.fips == 6111.0].mean(), color='purple', label='Ventura County Mean Value: 408k')

    plt.xlabel("Tax Value")
    plt.ylabel("Count")
    plt.title("Tax Values for each County")
    plt.ticklabel_format(style='plain')
    plt.legend()
    plt.show()
    
def pred_act(y_validate):
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.tax_value, color='purple', alpha=0.3, label="Actual Tax Value", bins=150)
    #plt.hist(y_validate.pred_lm, color='blue', alpha=0.3, label="3rd Degree Polynomial Model", bins=150)
    plt.hist(y_validate.pred_lars, color='teal', alpha=0.3, label="Lasso+Lars", bins=150)
    plt.hist(y_validate.pred_glm, color='black', alpha=0.2, label="Tweedie", bins=150)
    plt.xlabel("Tax Value")
    plt.ylabel("Count")
    plt.title("Comparing the Distribution of Actual Tax Value to Predicted Tax Value")
    plt.ticklabel_format(style='plain')
    plt.legend()
    plt.show()

def no_error(y_validate):
    # Plot
    plt.figure(figsize=(16,5))
    plt.scatter(y_validate.tax_value, y_validate.pred_lm, 
                alpha=0.5, color="orange", s=100, label="3rd degree Polynomial")
    plt.plot(y_validate.tax_value, y_validate.pred_mean, alpha=.3, color="black", label='_nolegend_')
    plt.plot(y_validate.tax_value, y_validate.tax_value, alpha=.5, color="blue")
    plt.scatter(y_validate.tax_value, y_validate.pred_glm, 
                 alpha=.2, color="black", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate.pred_lars, 
                alpha=.1, color="blue", s=100, label="Lasso-Lars")
    plt.ticklabel_format(style='plain')

    plt.legend()
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.show()
    
def vis(df, col, col2):
    plt.title(f'Relationship of {col2} and {col}')
    sns.barplot(x=col, y=col2, data=df)
    sns.barplot(x=col, y=col2, data=df).axhline(df.tax_value.mean())
    plt.show()
    
def graph(df, col1, col2):
    plt.hist(df.bedrooms, color='blue', alpha=0.4, bins=20)
    plt.hist(df.bathrooms, color='orange', alpha=0.4, bins = 20)

    plt.show()