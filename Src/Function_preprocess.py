import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

def preprocess(df_, chx_target:bool, chx_smote:bool):
    """
    Preprocesses the dataset.
    
    This function performs the following preprocessing steps on the dataset:
    1. Drops rows where the 'Failure Type' column is equal to 'Random Failures'.
    2. Drops rows where the 'Failure Type' column is equal to 'No Failure' and the 'Target' column is equal to 1.
    3. Memorizes the columns where the data type is 'object' (categorical) and removes the 'Failure Type' column from the list.
    4. Memorizes the columns where the data type is not 'object' (numerical) and removes the 'Target' column from the list.
    5. Drops the 'Target' column from the dataset and renames the 'Failure Type' column to 'Target'.
    6. Creates a ColumnTransformer object to apply one-hot encoding to the categorical columns and standard scaling to the numerical columns.
    7. Splits the dataset into training, validation, and test sets.
    8. Fits and transforms the datasets using the ColumnTransformer.
    9. Creates a dictionary to map the target values to their corresponding indices.
    10. Saves the validation dataset to a CSV file.
    11. Applies SMOTE (Synthetic Minority Over-sampling Technique) to the training and test datasets.
    12. Saves the resampled training and test datasets to CSV files.
    
    Parameters:
    - df_: The input dataset to be preprocessed.
    - chx_target: A boolean value indicating whether to check the 'Target' column for preprocessing.
    - chx_smote: A boolean value indicating whether to apply SMOTE to the training and test datasets.
    
    Returns:
    - df_train: The preprocessed training dataset.
    - df_test: The preprocessed test dataset.
    - dict_target_inv_: A dictionary mapping the target values to their corresponding indices.
    """

    # Drop rows where 'Failure Type' column is equal to 'Random Failures'
    df_ = df_[df_['Failure Type'] != 'Random Failures']

    # Drop rows where 'Failure Type' column is equal to 'No Failure' and 'Target' column is equal to 1
    df_ = df_[(df_['Failure Type'] != 'No Failure') | (df_['Target'] != 1)]

    # Memorize the columns where the data type is 'object' (categorical)
    cat_cols = df_.select_dtypes(include='O').columns.to_list()
    cat_cols.remove('Failure Type')

    # Memorize the columns where the data type is not 'object' (numerical)
    num_cols = df_.select_dtypes(exclude='O').columns.to_list()
    num_cols.remove('Target')

    if chx_target:
        # Drop the 'Target' column and rename the 'Failure Type' column to 'Target'
        df_.drop(columns=['Target'], inplace=True)
        df_.rename(columns={'Failure Type': 'Target'}, inplace=True)
    else:
        # Drop the 'Failure Type' column
        df_.drop(columns=['Failure Type'], inplace=True)

    # Create a ColumnTransformer object
    ct_X_ = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), cat_cols),
            ("num", StandardScaler(), num_cols)
            #("fail", OrdinalEncoder(), ['Target'])
        ],
        remainder='passthrough')
    
    if chx_target:
        ct_y_ = ColumnTransformer(
            transformers=[
                ("fail", OrdinalEncoder(), ['Target'])
            ],
            remainder='passthrough')
    else:
        df_['Target'] = df_['Target'].replace({1: 'With_Failure', 0: 'No Failure'})
        ct_y_ = ColumnTransformer(
            transformers=[
                ("fail", OrdinalEncoder(), ['Target'])
            ],
            remainder='passthrough')

    # Create different dataset for X train, test and validation

    # Create dataset for training and validation
    X_, X_val, y_, y_val = train_test_split(
        df_.drop(columns=['Target']), 
        df_['Target'], 
        test_size=0.2,
        stratify=df_['Target'],
        random_state=42)

    # Create dataset for training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_, 
        y_, 
        test_size=0.2,
        stratify=y_,
        random_state=42)

    # Fit and transform datasets
    X_train_trans, X_test_trans, X_val_trans = fit_transform(ct_X_, X_train, X_test, X_val)
    y_train_trans, y_test_trans, y_val_trans = fit_transform(ct_y_, pd.DataFrame(y_train), pd.DataFrame(y_test), pd.DataFrame(y_val))

    # Create a dictionary to map the target values to their corresponding indices
    cat_=ct_y_.named_transformers_['fail'].categories_
    dict_target_ = {}
    dict_target_inv_ = {}
    for i, c in enumerate(cat_[0]):
        dict_target_[c] = i
        dict_target_inv_[i] = c

    # Save the validation dataset
    #pd.concat([X_val_trans, y_val_trans], axis=1).to_csv('../data/predictive_maintenance_validation.csv')
    create_df_from_X_y(X_=X_val_trans, y_=y_val_trans, columns_y=dict_target_inv_.values()).to_csv('../data/predictive_maintenance_validation.csv', index=False)

    # Apply SMOTE to the dataset train and test
    if chx_smote:
        X_train_res_, y_train_res_ = apply_smote(X_=X_train_trans, y_=y_train_trans)
        X_test_res_, y_test_res_ = apply_smote(X_=X_test_trans, y_=y_test_trans)
    else:
        X_train_res_ = X_train_trans.copy()
        y_train_res_ = y_train_trans.copy()
        X_test_res_ = X_test_trans.copy()
        y_test_res_ = y_test_trans.copy()

    # Save the train and test dataset
    df_train = create_df_from_X_y(X_=X_train_res_, y_=y_train_res_, columns_y=dict_target_inv_.values())
    df_train.to_csv('../data/predictive_maintenance_train_resampled.csv', index=False)
    df_test = create_df_from_X_y(X_=X_test_res_, y_=y_test_res_, columns_y=dict_target_inv_.values())
    df_test.to_csv('../data/predictive_maintenance_test_resampled.csv', index=False)


    return df_train, df_test, dict_target_inv_

def fit_transform(ct_, train, test, val):

    # Fit and transform the train_ dataset
    train_trans_ = ct_.fit_transform(train)    # train_trans_ is a numpy array afetr this step
    train_trans_ = pd.DataFrame(train_trans_)
    train_trans_.columns = ct_.get_feature_names_out()

    # Transform the test_ dataset
    test_trans_ = ct_.transform(test)
    test_trans_ = pd.DataFrame(test_trans_)
    test_trans_.columns = ct_.get_feature_names_out()

    # Transform the val_ dataset
    val_trans_ = ct_.transform(val)
    val_trans_ = pd.DataFrame(val_trans_)
    val_trans_.columns = ct_.get_feature_names_out()

    return train_trans_, test_trans_, val_trans_

def apply_smote(X_,y_):
    smote = SMOTE(random_state=12)
    X_resampled_, y_resampled_ = smote.fit_resample(X_.to_numpy(), y_.to_numpy())

    X_resampled_= pd.DataFrame(X_resampled_)
    X_resampled_.columns = X_.columns
    y_resampled_ = pd.DataFrame(y_resampled_)
    y_resampled_.columns = y_.columns

    return X_resampled_, y_resampled_

def create_df_from_X_y(X_, y_, columns_y):

    y_ = pd.DataFrame(pd.get_dummies(y_.astype(str) ))
    y_.columns=list(columns_y)

    return pd.concat([X_, y_], axis=1)


    
    


