from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('model.pkl')
brand_encoder = joblib.load('brand_encoder.pkl')
model_encoder = joblib.load('model_encoder.pkl')
fuel_encoder = joblib.load('fuel_encoder.pkl')
location_encoder = joblib.load('location_encoder.pkl')
type_encoder = joblib.load('type_encoder.pkl')

owner_mapping = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth & above': 4
}

def preprocess_input(data):
    brand = data['input'][0].lower() if data['input'][0] else np.nan
    model_name = data['input'][1].lower() if data['input'][1] else np.nan
    transmission = data['input'][2].lower() if data['input'][2] else np.nan
    age = 2024 - data['input'][3] if data['input'][3] else np.nan
    fuel = data['input'][4].lower() if data['input'][4] else np.nan
    engine = int(data['input'][5]) if data['input'][5] else np.nan
    km = int(data['input'][6]) if data['input'][6] else np.nan
    owner = data['input'][7] if data['input'][7] else np.nan
    location = data['input'][8].lower() if data['input'][8] else np.nan
    mileage = float(data['input'][9]) if data['input'][9] else np.nan
    power = float(data['input'][10]) if data['input'][10] else np.nan
    seats = int(data['input'][11]) if data['input'][11] else np.nan
    car_type = data['input'][12].lower() if data['input'][12] else np.nan

    transmission = 0 if transmission == 'manual' else 1 if transmission == 'automatic' else np.nan

    if isinstance(owner, str):
        owner = owner_mapping.get(owner.lower(), np.nan)

    brand_encoded = brand_encoder.transform([brand])[0] if brand in brand_encoder.classes_ else np.nan
    model_encoded = model_encoder.transform([model_name])[0] if model_name in model_encoder.classes_ else np.nan
    fuel_encoded = fuel_encoder.transform([fuel])[0] if fuel in fuel_encoder.classes_ else np.nan
    location_encoded = location_encoder.transform([location])[0] if location in location_encoder.classes_ else np.nan
    car_type_encoded = type_encoder.transform([car_type])[0] if car_type in type_encoder.classes_ else np.nan

    input_data = [
        brand_encoded,
        model_encoded,
        transmission,
        age,
        fuel_encoded,
        engine,
        km,
        owner,
        location_encoded,
        mileage,
        power,
        seats,
        car_type_encoded
    ]
    
    return np.array(input_data).reshape(1, -1)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = preprocess_input(data)
    prediction = np.round(model.predict(input_data), -3)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)