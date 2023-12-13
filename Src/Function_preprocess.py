import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

def preprocess(df_):
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

    df_.drop(columns=['Target'], inplace=True)
    df_.rename(columns={'Failure Type': 'Target'}, inplace=True)

    # Create a ColumnTransformer object
    ct_ = ColumnTransformer([
        ("onehot", OneHotEncoder(), cat_cols),
        ("scale", StandardScaler(), num_cols),
        ("fail_type", OrdinalEncoder(), ['Target'])
    ])

    # Fit and transform the DataFrame
    df_ = ct_.fit_transform(df_)
    df_ = pd.DataFrame(df_)
    df_.columns = ct_.get_feature_names_out()

    # Create a dictionary to map the target values to their corresponding indices
    cat_=ct_.named_transformers_['fail_type'].categories_
    dict_target_ = {}
    dict_target_inv_ = {}
    for i, c in enumerate(cat_[0]):
        dict_target_[c] = i
        dict_target_inv_[i] = c

    # Create dataset for training and validation
    X_, X_val, y_, y_val = train_test_split(
        df_.drop(columns=['fail_type__Target']), 
        df_['fail_type__Target'], 
        test_size=0.2,
        stratify=df_['fail_type__Target'],
        random_state=42)
    
    # Save the validation dataset
    pd.concat([X_val, y_val], axis=1).to_csv('../data/predictive_maintenance_validation.csv')

    # Apply SMOTE to the training dataset
    smote = SMOTE(random_state=12)
    X_resampled_, y_resampled_ = smote.fit_resample(X_.to_numpy(), y_.to_numpy())
    
    # Create a DataFrame from the resampled data
    df_ = pd.DataFrame(X_resampled_, columns=X_.columns)
    y_resampled_ = pd.DataFrame(pd.get_dummies(y_resampled_))
    y_resampled_.columns=list(dict_target_inv_.values())
    df_ = pd.concat([df_, y_resampled_], axis=1)

    # Save the training dataset
    df_.to_csv('../data/predictive_maintenance_train.csv', index=False)

    return df_, dict_target_inv_


