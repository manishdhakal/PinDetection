# PinDetection Project
This repository contains a Python Flask backend for URL analysis and an Android client application. The backend is managed using uv with Python 3.9, and the Android app collects data/interacts with the API within the local network.

## 📂 Project Structure
```.
├── detector/
│   ├── api.py              # Flask app
│   └── requirements.txt    # Dependencies
└── app/                    # Android
    └── src/
        └── main/
            └── java/
                └── com/
                    └── example/
                        └── pindetection/
                            └── MainActivity.kt
```

## 🚀 Backend Setup
### Prerequisites
- Python 3.9
- uv (Python package manager)

### Installation

1. Initialize the virtual environment:

```bash
uv venv --python 3.9
```
2. Activate the environment
`source .venv/bin/activate`
3. Install dependencies: Point `uv` to the requirements file inside the detector folder.
```bash
uv pip install -r detector/requirements.txt
```

### Running the Server
To allow the Android device (which is an external device on the network) to connect, the Flask app must listen on `0.0.0.0`.

```bash
python detector/api.py
```

## 📱 Android Client Configuration
The Android app needs to know the **Local IP Address** of the computer running the Flask server to communicate within the same Wi-Fi network. 
1. You can clearly find **Local IP Address**  in the terminal of running Flask app.

2. Update `MainActivity.kt`. Replace the IP address with your computer's Local IP found in step 1.

```kt
// app/src/main/java/com/example/pindetection/MainActivity.kt

class MainActivity : AppCompatActivity() {

    // TODO: Update with your computer's Local IP (e.g., 192.168.1.15)
    // Do NOT use "localhost" or "127.0.0.1" for physical devices
    private val url = "<URL>/predict" 

    override fun onCreate(savedInstanceState: Bundle?) {
        // ...
        // ...
    }
}
```
## Network Troubleshooting
 - Ensure both devices are on the exact same Wi-Fi.
 - Ensure your computer's firewall is allowing traffic on Port `8000`.