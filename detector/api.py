from flask import Flask, request, jsonify
import torch
from cnn_model import CNN_Model

app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "API Working!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(data['passcode'])
        # print(data["sensor_history"], 'FHDFADSFDSKFHSD')
        keys = [
            "accX",
            "accY",
            "accZ",
            "gyroX",
            "gyroY",
            "gyroZ",
            "rotX",
            "rotY",
            "rotZ",
            "magX",
            "magY",
            "magZ",
        ]
        

        # Convert the list of dict to 2d array
        sensor_data = [[entry[key] for key in keys] for entry in data["sensor_history"]]
        
        sensor_data_tensor = torch.tensor(sensor_data).reshape(-1, 100, 12).float()

        model = load_model("/Users/snowman/Desktop/codes/PinDetect/detector/cnn_model.pth")
        with torch.no_grad():
            outputs = model(sensor_data_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_pins = predicted.tolist()
            
        result = {
            "status": "OK",
            "received_data": data,
            "prediction": predicted_pins,
        }

        print(f'Predicted pins is: {predicted_pins} for user pin is {data["passcode"]}')

        return jsonify(result), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
def load_model(model_path: str):
    model = CNN_Model(input_channels=12, num_classes=10)  # Adjust num_classes as needed
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
