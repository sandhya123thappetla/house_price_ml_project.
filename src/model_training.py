import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

from data_preprocessing import load_data, preprocess_data

def train_model():
    # Load and preprocess data
    df = load_data()
    X, y, preprocessor = preprocess_data(df)

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create full pipeline: preprocessing + model
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)
    print("Model trained successfully!")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"R² Score: {r2:.3f}")

    # Save the trained model
    with open('models/house_price_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved to models/house_price_model.pkl")

    return pipeline, mae, r2

# Run training if script executed
if __name__ == "__main__":
    train_model()