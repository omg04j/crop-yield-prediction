from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load the trained pipeline model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs
        region = request.form['Region']
        soil = request.form['Soil_Type']
        crop = request.form['Crop']
        rainfall = float(request.form['Rainfall_mm'])
        temperature = float(request.form['Temperature_Celsius'])
        fertilizer = request.form['Fertilizer_Used'] == 'True'
        irrigation = request.form['Irrigation_Used'] == 'True'
        weather = request.form['Weather_Condition']
        days = int(request.form['Days_to_Harvest'])

        # Create a DataFrame with the same column names as used in training
        input_df = pd.DataFrame([{
            'Region': region,
            'Soil_Type': soil,
            'Crop': crop,
            'Rainfall_mm': rainfall,
            'Temperature_Celsius': temperature,
            'Fertilizer_Used': fertilizer,
            'Irrigation_Used': irrigation,
            'Weather_Condition': weather,
            'Days_to_Harvest': days
        }])

        # Predict using the loaded pipeline
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=f"{prediction} tons/hectare")
    
    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template('index.html', prediction="Prediction failed. Check logs.")

if __name__ == '__main__':
    app.run(debug=True)
