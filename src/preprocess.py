from dataload import read_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("preprocess.py loaded")

def preprocess_data(df):
    df['Diabetes_012'] =df['Diabetes_012'].replace({2:1})
    df = df.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
    target = 'Diabetes_binary'
    independent_vars = df.drop(target, axis=1)
    dependent_var = df[target]
    df.head(5)
    
    return independent_vars, dependent_var

def train_test_data(X, y, test_size=0.1, random_state=25):
   

    scaler = StandardScaler()

    #Fit only on training data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_feature_names():
    """Return the list of independent variable names used for training."""
    df = read_data()
    X, _ = preprocess_data(df)
    return list(X.columns)


def build_scaler(X=None):
    """Return a StandardScaler fitted on X (or on full training data if X is None)."""
    if X is None:
        df = read_data()
        X, _ = preprocess_data(df)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_user_input(user_df):
    """
    Given a single-row (or multi-row) DataFrame with feature columns, reorder/align
    to training columns, fill missing columns with zeros, and return scaled numpy array.
    This fits a StandardScaler on the full training independent variables
    """
    # Ensure columns and ordering match training
    df_train = read_data()
    X_train, _ = preprocess_data(df_train)
    train_cols = list(X_train.columns)

    # Reindex user_df to the training columns; missing columns will be filled with
    # sensible defaults (training medians) rather than zeros to avoid biasing predictions.
    user_aligned = user_df.reindex(columns=train_cols)

    # Compute training medians for imputation
    medians = X_train.median()
    # Fill missing user values (NaN) using training medians
    user_aligned = user_aligned.fillna(medians)

    scaler = build_scaler(X_train)
    X_scaled = scaler.transform(user_aligned)
    return X_scaled

