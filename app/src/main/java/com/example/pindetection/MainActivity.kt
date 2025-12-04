package com.example.pindetection


import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.pindetection.ui.theme.PinDetectionTheme
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


class MainActivity : ComponentActivity(), SensorEventListener {

    // --- Sensor Variables ---
    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVector: Sensor? = null
    private var magneticField: Sensor? = null

    // REMOVED: isCollecting flag is no longer needed as collection starts automatically.

    // *** State to control 100-sample collection per tap (REQUIRED) ***
    private var isRecordingTap = false
    private var samplesToCollect = 0
    private var currentDigit = ""

    // Current sensor readings
    private var accX = 0f; private var accY = 0f; private var accZ = 0f
    private var gyroX = 0f; private var gyroY = 0f; private var gyroZ = 0f
    private var rotX = 0f; private var rotY = 0f; private var rotZ = 0f
    private var magX = 0f; private var magY = 0f; private var magZ = 0f

    data class SensorData(
        val accX: Float, val accY: Float, val accZ: Float,
        val gyroX: Float, val gyroY: Float, val gyroZ: Float,
        val rotX: Float, val rotY: Float, val rotZ: Float,
        val magX: Float, val magY: Float, val magZ: Float,
        val digit: String // optional, to know which digit was pressed
    )

    // Mutable list to store the sensor data
    val sensorDataList = mutableListOf<SensorData>()


    private var url="http://10.0.0.12:8000/predict"
    private var POST="POST";
    private var GET="GET";

    // Permission launcher for Android 9 and below
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            copyCsvToDownloads()
        } else {
            Toast.makeText(this, "Storage permission denied", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // --- 1. Setup Sensors ---
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        magneticField = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // --- 2. Initialize CSV File ---
        initializeCsvFile()

        // --- 3. AUTO-START SENSOR COLLECTION (No Start Button Needed) ---
        // Sensors are registered here and will run continuously
        registerSensorListeners()

        setContent {
            PinDetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // --- 4. Pass the click handler to the UI (Removed Start/Stop callbacks) ---
                    NumberPadScreen(
                        modifier = Modifier.padding(innerPadding),
                        onNumberClick = { number, onReset ->
                            handleZoneClick(number, onReset)
                        },
                        onOkClick = { currentPasscode, onUiReset ->
                            performApiCall(currentPasscode)
                            onUiReset()
                        }
                    )
                }
            }
        }
    }

    // New function to handle sensor registration
    private fun registerSensorListeners() {
        val delay = SensorManager.SENSOR_DELAY_GAME // ~50Hz
        accelerometer?.let { sensorManager.registerListener(this, it, delay) }
        gyroscope?.let { sensorManager.registerListener(this, it, delay) }
        rotationVector?.let { sensorManager.registerListener(this, it, delay) }
        magneticField?.let { sensorManager.registerListener(this, it, delay) }
        Log.d("MainActivity", "Sensor Stream Started Automatically at ${delay} delay.")
    }


    // --- Logic: Handle Button Click ---
    private fun handleZoneClick(number: String, onReset: () -> Unit) {

        // Prevent starting a new tap collection if one is already in progress
        if (isRecordingTap) {
            Toast.makeText(this, "Still collecting previous tap data. Please wait.", Toast.LENGTH_SHORT).show()
            return
        }

        // *** START 100-SAMPLE COLLECTION ***
        isRecordingTap = true
        samplesToCollect = 100 // Initialize counter for 100 samples
        currentDigit = number // Store the digit that was pressed

        // Notify user that collection has started (takes ~2 seconds at 50Hz)
        Toast.makeText(this, "Collecting $samplesToCollect samples for digit $number...", Toast.LENGTH_SHORT).show()

        // NOTE: The actual data collection and logging now happens in onSensorChanged,
        // which runs repeatedly at 50Hz until samplesToCollect hits zero.
    }

    // --- Logic: API Call Placeholder ---
    private fun performApiCall(passcode: String) {

        Thread {
            try {
                val client = OkHttpClient()
                val dataArray = org.json.JSONArray()

                // sensorDataList now contains N * 100 samples (where N is passcode length)
                sensorDataList.forEach { data ->
                    val item = JSONObject()
                    item.put("accX", data.accX)
                    item.put("accY", data.accY)
                    item.put("accZ", data.accZ)
                    item.put("gyroX", data.gyroX)
                    item.put("gyroY", data.gyroY)
                    item.put("gyroZ", data.gyroZ)
                    item.put("rotX", data.rotX)
                    item.put("rotY", data.rotY)
                    item.put("rotZ", data.rotZ)
                    item.put("magX", data.magX)
                    item.put("magY", data.magY)
                    item.put("magZ", data.magZ)
                    item.put("digit", data.digit)
                    dataArray.put(item)
                }

                val finalPayload = JSONObject()
                finalPayload.put("passcode", passcode)
                finalPayload.put("sensor_history", dataArray)

                val requestBody = finalPayload.toString().toRequestBody("application/json; charset=utf-8".toMediaType())


                val request = okhttp3.Request.Builder()
                    .url("http://10.0.0.12:8000/predict")
                    .post(requestBody)
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string()

                    if (response.isSuccessful) {
                        runOnUiThread {
                            // Display success message and server response
                            Toast.makeText(this, "API Success! Passcode: $passcode. Response: ${responseBody?.take(50)}...", Toast.LENGTH_LONG).show()
                        }
                    } else {
                        runOnUiThread {
                            // Display error message
                            Toast.makeText(this, "API Failed. Code: ${response.code}. Response: ${responseBody?.take(50)}...", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "API Call Error: ${e.message}", e)
                runOnUiThread {
                    Toast.makeText(this, "Network Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally{
                runOnUiThread { sensorDataList.clear() }
            }
        }.start()

        Log.d("MainActivity", "--- API Call Triggered ---")


        // *** CHANGE: Add the passcode to the Toast message ***
        val toastMessage = if (passcode.isEmpty()) {
            "API Call Executed. Passcode was empty."
        } else {
            "API Call Executed for Passcode: $passcode: $url"
        }

        Toast.makeText(this, toastMessage, Toast.LENGTH_LONG).show()
    }


    // --- Logic: File Operations ---
    private fun initializeCsvFile() {
        val csvFile = File(filesDir, "dataset.txt")
        if (!csvFile.exists()) {
            try {
                if (filesDir.isDirectory && filesDir.canWrite()) {
                    csvFile.writeText(
                        // Timestamp is recorded here (Requirement b)
                        "Timestamp,Zone,AccX,AccY,AccZ,GyroX,GyroY,GyroZ,RotX,RotY,RotZ,MagX,MagY,MagZ\n"
                    )
                    Log.d("MainActivity", "Created dataset.txt at ${csvFile.absolutePath}")
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error creating dataset.txt: ${e.message}")
            }
        }
    }

    private fun appendToCsv(zoneIndex: Int) {
        if (filesDir.isDirectory && filesDir.canWrite()) {
            val csvFile = File(filesDir, "dataset.txt")
            // Capturing timestamp with millisecond precision (Requirement b)
            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
            val csvRow = "$timestamp,$zoneIndex,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n".format(
                accX, accY, accZ, gyroX, gyroY, gyroZ, rotX, rotY, rotZ, magX, magY, magZ
            )
            try {
                FileOutputStream(csvFile, true).use { it.write(csvRow.toByteArray()) }
                // Logging every row is too chatty, so only log completion
                // Log.d("MainActivity", "Appended data: $csvRow")
                checkAndCopyCsvToDownloads()
            } catch (e: Exception) {
                Log.e("MainActivity", "Error writing to CSV: ${e.message}")
            }
        }
    }

    private fun checkAndCopyCsvToDownloads() {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                copyCsvToDownloads()
            } else {
                requestPermissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        } else {
            copyCsvToDownloads()
        }
    }

    private fun copyCsvToDownloads() {
        try {
            val internalCsvFile = File(filesDir, "dataset.txt")
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (downloadsDir.isDirectory && downloadsDir.canWrite()) {
                val externalCsvFile = File(downloadsDir, "dataset.txt")
                internalCsvFile.copyTo(externalCsvFile, overwrite = true)
                Log.d("MainActivity", "Copied to Downloads: ${externalCsvFile.absolutePath}")
                Toast.makeText(this, "Data Saved to Downloads/dataset.txt", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying to Downloads: ${e.message}")
        }
    }

    // --- Sensor Lifecycle Methods ---
    override fun onResume() {
        super.onResume()
        // Ensure sensors are registered when the app comes to the foreground
        registerSensorListeners()
    }

    override fun onPause() {
        super.onPause()
        // Unregister listeners to save battery when the app is paused
        sensorManager.unregisterListener(this)
        Log.d("MainActivity", "Sensor Stream Stopped on Pause.")
        // Stop any ongoing tap recording
        isRecordingTap = false
        samplesToCollect = 0
        currentDigit = ""
    }

    override fun onSensorChanged(event: SensorEvent) {
        // Update current sensor values
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { accX = event.values[0]; accY = event.values[1]; accZ = event.values[2] }
            Sensor.TYPE_GYROSCOPE -> { gyroX = event.values[0]; gyroY = event.values[1]; gyroZ = event.values[2] }
            Sensor.TYPE_ROTATION_VECTOR -> { rotX = event.values[0]; rotY = event.values[1]; rotZ = event.values[2] }
            Sensor.TYPE_MAGNETIC_FIELD -> { magX = event.values[0]; magY = event.values[1]; magZ = event.values[2] }
        }

        // *** CORE LOGIC FOR 100 SAMPLES PER TAP ***
        if (isRecordingTap && samplesToCollect > 0) {
            val zoneIndex = currentDigit.toIntOrNull() ?: -1

            val currentSensorData = SensorData(
                accX = accX, accY = accY, accZ = accZ,
                gyroX = gyroX, gyroY = gyroY, gyroZ = gyroZ,
                rotX = rotX, rotY = rotY, rotZ = rotZ,
                magX = magX, magY = magY, magZ = magZ,
                digit = currentDigit
            )

            // 1. Add sample to the list for the API call
            sensorDataList.add(currentSensorData)

            // 2. Log sample to CSV file
            appendToCsv(zoneIndex)

            // 3. Decrement counter
            samplesToCollect--

            // 4. Check if collection is complete
            if (samplesToCollect == 0) {
                isRecordingTap = false
                Log.d("MainActivity", "--- 100 samples collected for digit $currentDigit ---")
                // Use runOnUiThread because this runs on the sensor thread
                runOnUiThread {
                    Toast.makeText(this, "100 samples recorded for $currentDigit.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}

// --- Composable UI (START/STOP buttons removed) ---

@Composable
fun NumberPadScreen(
    modifier: Modifier = Modifier,
    onNumberClick: (String, () -> Unit) -> Unit,
    onOkClick: (String, () -> Unit) -> Unit,
    // REMOVED: onStartClick and onStopClick arguments
) {
    // Visual state for the asterisks
    var passcode by remember { mutableStateOf("") }
    // To know passcodeValue
    var passcodeVal by remember { mutableStateOf("") }


    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color.White),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        // REMOVED: Start/Stop Buttons Row

        Text(
            text = if (passcode.isEmpty()) "Enter Passcode" else passcode,
            color = Color.Black,
            fontSize = 22.sp,
            modifier = Modifier.padding(bottom = 48.dp),
            maxLines = 1, // Keep to single line
            softWrap = false, // Disable text wrapping
            overflow = TextOverflow.Visible // Allow overflow without adding rows
        )

        // Wrapper to handle local UI update + callback
        val handleDigitClick: (String) -> Unit = { digit ->
            // Update UI immediately
            passcode += "*"
            passcodeVal += digit

            // Trigger 100-sample collection in MainActivity
            onNumberClick(digit) {
                // This reset block runs when the dialog is dismissed (but no dialog now)
                passcode = ""
                passcodeVal =""
            }
        }

        // Pass the wrapper down to the rows
        NumberPadRow(listOf("1", "2", "3"), handleDigitClick)
        Spacer(modifier = Modifier.height(24.dp))

        NumberPadRow(listOf("4", "5", "6"), handleDigitClick)
        Spacer(modifier = Modifier.height(24.dp))

        NumberPadRow(listOf("7", "8", "9"), handleDigitClick)
        Spacer(modifier = Modifier.height(24.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            // OK button
            TextButton(
                label = "OK",
                onClick = {
                    onOkClick(passcodeVal) {
                        passcode = ""
                        passcodeVal = ""
                    }
                }
            )
            // 0 button
            NumberButton(number = "0", onClick = handleDigitClick)

            // Backspace Button
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(80.dp)
                    .clickable {
                        if (passcode.isNotEmpty()) {
                            passcode = passcode.dropLast(1)
                            passcodeVal = passcodeVal.dropLast(1) // Also remove from value
                        }
                    }
            ) {
                Text(
                    text = "⌫",
                    color = Color.Black,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun NumberPadRow(
    numbers: List<String>,
    onNumberClick: (String) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        for (number in numbers) {
            NumberButton(number = number, onClick = onNumberClick)
        }
    }
}

@Composable
fun NumberButton(
    number: String,
    onClick: (String) -> Unit
) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(80.dp)
            .clip(CircleShape)
            .background(Color(0xFFF2F2F7)) // Light Gray Background
            .clickable { onClick(number) }
    ) {
        Text(
            text = number,
            color = Color.Black,
            fontSize = 36.sp,
            fontWeight = FontWeight.Thin
        )
    }
}

@Preview(showBackground = true)
@Composable
fun NumberPadPreview() {
    PinDetectionTheme {
        // Updated preview to match new signature
        NumberPadScreen(
            onNumberClick = { _, _ -> },
            onOkClick = { _, _ -> }
            // Removed onStartClick and onStopClick
        )
    }
}

@Composable
fun TextButton(
    label: String,
    onClick: () -> Unit,
    color: Color = Color(0xFFE0E0E0)
) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(80.dp)
            .clip(CircleShape)
            // Use dynamic color for the background
            .background(color)
            // Changed onClick handler to simple () -> Unit
            .clickable { onClick() }
    ) {
        Text(
            text = label,
            color = Color.Black,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}