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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures

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

def wrangle_zillow(df):
    df = df[(df.propertylandusedesc == 'Single Family Residential') | (df.propertylandusedesc == 'Inferred Single Family Residential')]
    df.rename(columns={'bedroomcnt':'bedrooms',
                       'bathroomcnt':'bathrooms', 
                       'calculatedfinishedsquarefeet':'f_sqft', 
                       'taxvaluedollarcnt':'tax_value',
                       'taxamount':'tax_amt',}, inplace=True)
    # Create a new column for month of transaction
    df['month'] = df['transactiondate'].str[5:7]
    df['transactiondate'] = df['transactiondate'].str[:4]
    # Reassign the df to only the values where the house had a transaction in 2017
    df = df[df.transactiondate == '2017']
    df.bedrooms.fillna(random.randint(2.0, 5.0), inplace = True)
    df.bathrooms.fillna(random.randint(1.0, 3.0), inplace = True)
    df.f_sqft.fillna(df.f_sqft.median(), inplace = True)
    df.tax_value.fillna(df.tax_value.mode().max(), inplace = True)
    # For yearbuilt I'll use 1958 as it falls in the middle of the mean and mode and they are all fairly close in value
    df.yearbuilt.fillna(df.yearbuilt.median(), inplace = True)
    df.tax_amt.fillna(df.tax_amt.median(), inplace = True)
    return df

def scale_data(train, 
               validate, 
               test, 
               columns=['bedrooms', 'bathrooms', 'f_sqft', 'tax_amt'],
               return_scaler=False):
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

# This function compares churn against all other columns
def telco_vis(train, col):
    plt.title('Relationship of churn and '+col)
    sns.barplot(x=col, y='churn_Yes', data=train)
    sns.barplot(x=col, y='churn_Yes', data=train).axhline(train.churn_Yes.mean())
    plt.show()
    
def telco_analysis(train, col):
    telco_vis(train, col)
    test(train, 'churn_Yes', col)
    
# This function is all encompassing of telco_vis and telco_analysis, 
def telco_test(df):
    for col in df.columns.tolist():
        print(telco_analysis(df, col))
        print('-------')
        print(pd.crosstab(df.churn_Yes, df[col]))
        print('-------')
        print(stats.chi2_contingency(pd.crosstab(df.churn_Yes, df[col])))
        print('-------')
        print(df[col].value_counts())
        print(df[col].value_counts(normalize=True))
        
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
    
    return X_train.columns[kbest.get_support()],X_train_transformed.head(3)

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

'''Student Grade data'''
def wrangle_grades():
    """
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")
    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)
    # Drop all rows with NaN values.
    df = grades.dropna()
    # Convert all columns to int64 data types.
    df = df.astype("int")
    return df

# Generic helper function to provide connection url for Codeup database server.

def get_db_url(db_name):
    """
    This function uses my env file to get the url to access the Codeup database.
    It takes in a string identifying the database I want to connect to.
    """
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"


# Generic function that takes in a database name and a query.

def get_data_from_sql(str_db_name, query):
    """
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    """
    df = pd.read_sql(query, get_db_url(str_db_name))
    return df


# Mother function to acquire and prepare Telco data.

def wrangle_telco():
    """
    Queries the telco_churn database
    Returns a clean df with four columns:
    customer_id(object), monthly_charges(float), tenure(int), total_charges(float)
    """
    query = """
            SELECT
                customer_id,
                monthly_charges,
                tenure,
                total_charges
            FROM customers
            JOIN contract_types USING(contract_type_id)
            WHERE contract_type = 'Two year';
            """
    df = get_data_from_sql("telco_churn", query)

    # Replace any tenures of 0 with 1
    df.tenure = df.tenure.replace(0, 1)

    # Replace the blank total_charges with the monthly_charge for tenure == 1
    df.total_charges = np.where(
        df.total_charges == " ", df.monthly_charges, df.total_charges
    )

    # Convert total_charges to a float.
    df.total_charges = df.total_charges.astype(float)

    return df


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
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_numeric_X_cols(X_train, object_cols):
    """
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    return X_train_scaled, X_validate_scaled, X_test_scaled


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def create_dummies(df, object_cols):
    """
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns.
    It then appends the dummy variables to the original dataframe.
    It returns the original df with the appended dummy variables.
    """

    # run pd.get_dummies() to create dummy vars for the object columns.
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)

    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df


### student_mat.csv for feature engineering lesson
def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")

    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)

    # create dummy vars
    df = create_dummies(df, object_cols)

    # split data
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(
        df, "G3"
    )

    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(
        X_train, X_validate, X_test, numeric_cols
    )

    return (
        df,
        X_train,
        X_train_scaled,
        y_train,
        X_validate_scaled,
        y_validate,
        X_test_scaled,
        y_test,
    )

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
        
    return df