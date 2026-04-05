import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath='data/house_data.csv'):
    """Load CSV data into a DataFrame"""
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully! Shape:", df.shape)
    return df

def preprocess_data(df):
    """Prepare features and target, and define preprocessing pipeline"""
    # Drop Property_ID because it's just an identifier
    X = df.drop(['Property_ID', 'Price'], axis=1)
    y = df['Price']

    numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    categorical_features = ['Location', 'Property_Type']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    print("Preprocessing pipeline created!")
    return X, y, preprocessor

# Quick test
if __name__ == "__main__":
    df = load_data()
    X, y, preprocessor = preprocess_data(df)
    print("Features preview:\n", X.head())
    print("Target preview:\n", y.head())