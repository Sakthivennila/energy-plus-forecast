from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from energy_pulse_forecast import generate_dummy_data, train_and_evaluate_model
import pandas as pd
import os

app = Flask(__name__, static_folder='static')
CORS(app)

print("Initializing and training the model...")
df = generate_dummy_data(1000)
model = train_and_evaluate_model(df)
print("Model ready.")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        occupancy = float(data.get('occupancy', 0))
        temperature = float(data.get('temperature', 0))
        
        input_data = pd.DataFrame([[occupancy, temperature]], columns=['Occupancy', 'Temperature'])
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'occupancy': occupancy,
            'temperature': temperature
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
