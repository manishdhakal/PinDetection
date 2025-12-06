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

    // *** State to control 100-sample collection per tap ***
    private var isRecordingTap = false
    private var samplesToCollect = 0
    private var currentDigit = ""

    // *** Unique filename for this collection session ***
    private var csvFileName: String = ""

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
        val digit: String
    )

    // Mutable list to store the sensor data
    val sensorDataList = mutableListOf<SensorData>()

    private var url = "http://131.96.47.203:8000/predict"

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

        csvFileName = "sensor.txt"

        // --- 3. Initialize CSV File ---
        initializeCsvFile()

        // --- 4. AUTO-START SENSOR COLLECTION ---
        registerSensorListeners()

        setContent {
            PinDetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // --- 5. Pass the click handler to the UI ---
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

    // Function to handle sensor registration with specific sampling rate
    private fun registerSensorListeners() {
        // CHANGED: 5000 microseconds = 5ms = 200Hz
        val samplingPeriodUs = 5000

        accelerometer?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
        gyroscope?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
        rotationVector?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }
        magneticField?.let { sensorManager.registerListener(this, it, samplingPeriodUs) }

        Log.d("MainActivity", "Sensor Stream Started Automatically at ${samplingPeriodUs}us delay (200Hz).")
    }

    // --- Logic: Handle Button Click ---
    private fun handleZoneClick(number: String, onReset: () -> Unit) {

        // Prevent starting a new tap collection if one is already in progress
        if (isRecordingTap) {
            runOnUiThread {
//                Toast.makeText(this, "Still collecting previous tap data...", Toast.LENGTH_SHORT).show()
            }
            return
        }

        // *** START 100-SAMPLE COLLECTION ***
        isRecordingTap = true
        samplesToCollect = 100 // Record exactly 100 samples
        currentDigit = number // This digit will be used for all 100 rows

        runOnUiThread {
//            Toast.makeText(this, "Recording 100 samples for Zone $number...", Toast.LENGTH_SHORT).show()
        }
    }

    // --- Logic: API Call Placeholder ---
    private fun performApiCall(passcode: String) {
        Thread {
            try {
                val client = OkHttpClient()
                val dataArray = org.json.JSONArray()

                // Synchronize access to sensorDataList
                val snapshotList = synchronized(sensorDataList) {
                    ArrayList(sensorDataList)
                }

                snapshotList.forEach { data ->
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
                    .url(url)
                    .post(requestBody)
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string()
                    val message = if (response.isSuccessful) "API Success" else "API Failed: ${response.code}"
                    runOnUiThread {
                        Toast.makeText(this, "$message. Response: ${responseBody?.take(50)}...", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "API Call Error: ${e.message}", e)
                runOnUiThread {
                    Toast.makeText(this, "Network Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                runOnUiThread {
                    synchronized(sensorDataList) {
                        sensorDataList.clear()
                    }
                }
            }
        }.start()
    }

    // --- Logic: File Operations ---
    private fun initializeCsvFile() {
        val csvFile = File(filesDir, csvFileName)
        if (!csvFile.exists()) {
            try {
                if (filesDir.isDirectory && filesDir.canWrite()) {
                    csvFile.writeText(
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
            val csvFile = File(filesDir, csvFileName)
            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())

            // CHANGED: Increased precision to 6 decimal places (%.6f)
            val csvRow = "$timestamp,$zoneIndex,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n".format(
                accX, accY, accZ, gyroX, gyroY, gyroZ, rotX, rotY, rotZ, magX, magY, magZ
            )
            try {
                FileOutputStream(csvFile, true).use { it.write(csvRow.toByteArray()) }
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
            val internalCsvFile = File(filesDir, csvFileName)
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            if (downloadsDir.isDirectory && downloadsDir.canWrite()) {
                val externalCsvFile = File(downloadsDir, csvFileName)
                internalCsvFile.copyTo(externalCsvFile, overwrite = true)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying to Downloads: ${e.message}")
        }
    }

    // --- Sensor Lifecycle Methods ---
    override fun onResume() {
        super.onResume()
        registerSensorListeners()
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
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

            // 1. Add to memory list for API
            synchronized(sensorDataList) {
                sensorDataList.add(currentSensorData)
            }

            // 2. Write to CSV with the current digit as the Zone label
            appendToCsv(zoneIndex)

            // 3. Decrement counter
            samplesToCollect--

            // 4. Check completion
            if (samplesToCollect == 0) {
                isRecordingTap = false
                // Copy CSV to downloads only after the batch is finished
                checkAndCopyCsvToDownloads()
                runOnUiThread {
//                    Toast.makeText(this, "Collected 100 samples for Zone $currentDigit", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}

// --- Composable UI ---

@Composable
fun NumberPadScreen(
    modifier: Modifier = Modifier,
    onNumberClick: (String, () -> Unit) -> Unit,
    onOkClick: (String, () -> Unit) -> Unit,
) {
    var passcode by remember { mutableStateOf("") }
    var passcodeVal by remember { mutableStateOf("") }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Color.White),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = if (passcode.isEmpty()) "Enter Passcode" else passcode,
            color = Color.Black,
            fontSize = 22.sp,
            modifier = Modifier.padding(bottom = 48.dp),
            maxLines = 1,
            softWrap = false,
            overflow = TextOverflow.Visible
        )

        val handleDigitClick: (String) -> Unit = { digit ->
            passcode += "*"
            passcodeVal += digit

            // Trigger recording in MainActivity
            onNumberClick(digit) {
                passcode = ""
                passcodeVal = ""
            }
        }

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
            // Empty placeholder for alignment
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF34C759)) // Green Color
                    .clickable {
                        onOkClick(passcodeVal) {
                            passcode = ""
                            passcodeVal = ""
                        }
                    }
            ) {
                Text(
                    text = "OK",
                    color = Color.White,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            NumberButton(number = "0", onClick = handleDigitClick)

            // OK / Check Button
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(80.dp)
                    .clickable {
                        if (passcode.isNotEmpty()) {
                            passcode = passcode.dropLast(1)
                            passcodeVal = passcodeVal.dropLast(1)
                        }
                    }
            ) {
                Text(
                    text = "⌫", // Backspace symbol
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
            .background(Color(0xFFF2F2F7))
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
        NumberPadScreen(onNumberClick = { _, _ -> }, onOkClick = { _, _ -> })
    }
}
