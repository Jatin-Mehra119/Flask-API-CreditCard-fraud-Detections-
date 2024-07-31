import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load Model
model = joblib.load('Credit-card-Fraud-Model.pkl')

# Create Flask App
app = Flask(__name__)

# Define Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        """ 
        json structure:
        
        example
        
        {
        "Unnamed: 0.1": 0,
        "Unnamed: 0": 0,
        "trans_date_trans_time": "2019-01-01 00:00:18",
        "cc_num": "2703186189652095",
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "amt": 4.97,
        "first": "Jennifer",
        "last": "Banks",
        "gender": "F",
        "street": "561 Perry Cove",
        "city": "Moravian Falls",
        "state": "NC",
        "zip": "28654",
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 3495,
        "job": "Psychologist, counselling",
        "dob": "1988-03-09",
        "trans_num": "0b242abb623afc578575680df30655b9",
        "unix_time": 1325376018,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
        "is_fraud": 0
    }
        
        """
        
        # Get Data
        data = request.get_json(force=True)
        data_unseen = pd.DataFrame([data])
        
        # Make Prediction
        prediction = model.predict(data_unseen)
        
        # Return Prediction
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 400

# Run App
if __name__ == '__main__':
    app.run(debug=True)
