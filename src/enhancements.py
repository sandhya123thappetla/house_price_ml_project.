import pandas as pd
import pickle
import csv

# Load trained model
with open('models/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input validation function
def validate_input(data):
    for key, value in data.items():
        if value is None:
            raise ValueError(f"{key} is missing")
        if isinstance(value, (int, float)) and value < 0:
            raise ValueError(f"{key} cannot be negative")
    return True

# Predict function
def predict_price(model, data):
    validate_input(data)
    df = pd.DataFrame([data])
    price = model.predict(df)[0]
    return round(price, 2)

# Save prediction to CSV
def save_prediction(data, price, filename='predictions.csv'):
    data['Predicted_Price'] = price
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

# Feature importance (for Random Forest)
def show_feature_importance(model):
    importances = model.named_steps['regressor'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print("Feature Importance:")
    print(df.sort_values(by='Importance', ascending=False))

# Example new property
new_property = {
    'Area': 1200,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Age': 5,
    'Location': 'City Center',
    'Property_Type': 'Apartment'
}

try:
    price = predict_price(model, new_property)
    print(f"Predicted Price: ₹{price:,.2f}")
    save_prediction(new_property, price)
    show_feature_importance(model)
except Exception as e:
    print("Error:", e)