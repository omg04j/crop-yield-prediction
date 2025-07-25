from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Home route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        region = request.form['Region']
        soil = request.form['Soil_Type']
        crop = request.form['Crop']
        rainfall = float(request.form['Rainfall_mm'])
        temperature = float(request.form['Temperature_Celsius'])
        fertilizer = request.form['Fertilizer_Used'] == 'True'
        irrigation = request.form['Irrigation_Used'] == 'True'
        weather = request.form['Weather_Condition']
        days = int(request.form['Days_to_Harvest'])

        # Convert to DataFrame
        input_data = {
            'Region': [region],
            'Soil_Type': [soil],
            'Crop': [crop],
            'Rainfall_mm': [rainfall],
            'Temperature_Celsius': [temperature],
            'Fertilizer_Used': [fertilizer],
            'Irrigation_Used': [irrigation],
            'Weather_Condition': [weather],
            'Days_to_Harvest': [days]
        }
        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(input_df)
        return render_template('index.html', prediction_text=f"🌾 Predicted Yield: {prediction[0]:.2f} tons/hectare")

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)
