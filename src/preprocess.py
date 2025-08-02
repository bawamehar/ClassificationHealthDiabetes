from src.dataload import read_data
from sklearn.model_selection import train_test_split

print("preprocess.py loaded")

def bin_target(df):
    df['Diabetes_012'] =df['Diabetes_012'].replace({2:1})
    df = df.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
    return df

target = 'Diabetes_binary'

def separate_bin_num(df):
    binary_col = [
        col for col in df.columns
        if df[col].nunique() == 2 and col != target
    ]
    num_col = [
        col for col in df.columns.difference(binary_col) 
        if col != target
    ]
    return binary_col, num_col

def train_test_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=25)
    return X_train, X_test, y_train, y_test