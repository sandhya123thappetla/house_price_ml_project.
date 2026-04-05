import pickle
import pandas as pd

# Load trained model
def load_model(path='models/house_price_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model

# Predict function
def predict_price(model, data):
    """
    data: dictionary with keys:
    'Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location', 'Property_Type'
    """
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return round(prediction, 2)

# Quick test
if __name__ == "__main__":
    model = load_model()

    # Example new property
    new_property = {
        'Area': 1200,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age': 5,
        'Location': 'City Center',
        'Property_Type': 'Apartment'
    }

    price = predict_price(model, new_property)
    print(f"Predicted Price: ₹{price:,.2f}")
    