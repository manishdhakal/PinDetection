from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "API Working!"})


@app.route('/getname/<name>', methods=['POST'])
def extract_name(name):
    return "I got your name "+name

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(data, 'FHDFADSFDSKFHSD')

        if data is None:
            return jsonify({"error": "No JSON received"}), 400

        # Example: expected keys from Android
        required_fields = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]

        missing = [f for f in required_fields if f not in data['sensor_history']]
        
        
        if 'sensor_history' not in data.keys() or len(data['sensor_history']) != len(data['passcode']):
            return jsonify({"error": f"You did not provide sensor information for input"}), 400
            
        missing = False
        for sensor_field in data['sensor_history']:
            missing = [f for f in required_fields if f not in sensor_field.keys()]
            
            
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        
        #  Call your model here. if we are using MLP then save the trained model checkpoint and then call here.
        predicted_pins = '1245'
        
        
        
        result = {
            "status": "OK",
            "received_data": data,
            "prediction": 'predicted_pins'   # dummy output
        }
        
        print(f'Predicted pins is: {predicted_pins} for user pin is {data["passcode"]}')

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)