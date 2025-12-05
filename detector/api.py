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
        sensor_data_tensor = normalize_data(sensor_data_tensor)
        model = load_model("./detector/cnn_model.pth")
        with torch.no_grad():
            outputs = model(sensor_data_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_pins = predicted.tolist()

        result = {
            "status": "OK",
            "received_data": data,
            "prediction": predicted_pins,
        }

        print(
            f'Predicted pins: {"".join(map(str, predicted_pins))} | Ground truth pins: {"".join(data["passcode"])}'
        )

        return jsonify(result), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


def load_model(model_path: str):
    model = CNN_Model(input_channels=12, num_classes=10)  # Adjust num_classes as needed
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def normalize_data(data: torch.Tensor) -> torch.Tensor:
    mean = [ -1.1732,   4.5000,   7.9661,  -0.0931,  -0.0925,  -0.0563,   0.1788,
          0.1124,   0.1589,  -1.4378,  -4.2999, -28.1295]

    std = [ 2.6656,  1.7279,  2.0722,  0.4982,  0.9411,  0.4136,  0.1320,  0.1904,
         0.5978, 54.2114, 53.9228, 26.9856]
    
    mean = torch.tensor(mean).reshape(1, 1, -1)
    std = torch.tensor(std).reshape(1, 1, -1)
    return (data - mean) / ( std + 1e-7)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
