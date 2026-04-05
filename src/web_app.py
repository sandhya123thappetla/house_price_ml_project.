from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load trained model
with open('models/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'Area': float(request.form['Area']),
        'Bedrooms': int(request.form['Bedrooms']),
        'Bathrooms': int(request.form['Bathrooms']),
        'Age': int(request.form['Age']),
        'Location': request.form['Location'],
        'Property_Type': request.form['Property_Type']
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction_text=f"Predicted Price: ₹{prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)