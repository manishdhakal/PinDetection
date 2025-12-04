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
import okhttp3.FormBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody
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

        setContent {
            PinDetectionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // --- 3. Pass the click handler to the UI ---
                    NumberPadScreen(
                        modifier = Modifier.padding(innerPadding),
                        onNumberClick = { number, onReset ->
                            handleZoneClick(number, onReset)
                        },
                        onOkClick = { currentPasscode, onUiReset ->
                            performApiCall(currentPasscode)
                            onUiReset() // <--- EXECUTE THE UI RESET HERE
                        }
                    )
                }
            }
        }
    }

    // --- Logic: Handle Button Click ---
    private fun handleZoneClick(number: String, onReset: () -> Unit) {
        val zoneIndex = number.toIntOrNull() ?: return

        val currentSensorData = SensorData(
            accX = accX, accY = accY, accZ = accZ,
            gyroX = gyroX, gyroY = gyroY, gyroZ = gyroZ,
            rotX = rotX, rotY = rotY, rotZ = rotZ,
            magX = magX, magY = magY, magZ = magZ,
            digit = number
        )

        sensorDataList.add(currentSensorData)

        // Create and show dialog with sensor data
//        val dialogMessage = """
//            Zone $zoneIndex Sensor Data:
//            Accelerometer: X=%.2f, Y=%.2f, Z=%.2f
//            Gyroscope: X=%.2f, Y=%.2f, Z=%.2f
//            Rotation Vector: X=%.2f, Y=%.2f, Z=%.2f
//            Magnetic Field: X=%.2f, Y=%.2f, Z=%.2f
//        """.trimIndent().format(
//            accX, accY, accZ,
//            gyroX, gyroY, gyroZ,
//            rotX, rotY, rotZ,
//            magX, magY, magZ
//        )
//
//        AlertDialog.Builder(this)
//            .setTitle("Zone $zoneIndex Sensor Readings")
//            .setMessage(dialogMessage)
//            .setPositiveButton("OK") { dialog, _ ->
//                dialog.dismiss()
//                onReset() // Reset the UI dots here
//            }
//            .create()
//            .show()

        // Append sensor data to CSV
        appendToCsv(zoneIndex)
    }

    // --- Logic: API Call Placeholder ---
    private fun performApiCall(passcode: String) {

        Thread {
            try {
                val client = OkHttpClient()
                val json = JSONObject()
                val dataArray = org.json.JSONArray()

                sensorDataList.forEach { data ->
                    val item = JSONObject()
                    item.put("accX", data.accX)
                    item.put("accY", data.accY)
                    item.put("accZ", data.accZ)
                    item.put("gyroX", data.gyroX)
                    item.put("gyroY", data.gyroY)
                    item.put("gyroZ", data.gyroZ)
                    // Add other sensor data fields if needed, or only the relevant ones
                    item.put("digit", data.digit)
                    dataArray.put(item)
                }

                val finalPayload = JSONObject()
                finalPayload.put("passcode", passcode)
                finalPayload.put("sensor_history", dataArray)

                json.put("accX", accX)
                json.put("accY", accY)
                json.put("accZ", accZ)
                json.put("gyroX", gyroX)
                json.put("gyroY", gyroY)
                json.put("gyroZ", gyroZ)

                json.put("digit", passcode)

//                val requestBody = json.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
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
            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
            val csvRow = "$timestamp,$zoneIndex,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n".format(
                accX, accY, accZ, gyroX, gyroY, gyroZ, rotX, rotY, rotZ, magX, magY, magZ
            )
            try {
                FileOutputStream(csvFile, true).use { it.write(csvRow.toByteArray()) }
                Log.d("MainActivity", "Appended data: $csvRow")
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
//            if (downloadsDir.isDirectory && downloadsDir.canWrite()) {
//                val externalCsvFile = File(downloadsDir, "dataset.txt")
//                internalCsvFile.copyTo(externalCsvFile, overwrite = true)
//                Log.d("MainActivity", "Copied to Downloads: ${externalCsvFile.absolutePath}")
//                Toast.makeText(this, "Data Saved to Downloads/dataset.txt", Toast.LENGTH_SHORT).show()
//            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error copying to Downloads: ${e.message}")
        }
    }

    // --- Sensor Lifecycle Methods ---
    override fun onResume() {
        super.onResume()
        val delay = SensorManager.SENSOR_DELAY_GAME
        accelerometer?.let { sensorManager.registerListener(this, it, delay) }
        gyroscope?.let { sensorManager.registerListener(this, it, delay) }
        rotationVector?.let { sensorManager.registerListener(this, it, delay) }
        magneticField?.let { sensorManager.registerListener(this, it, delay) }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> { accX = event.values[0]; accY = event.values[1]; accZ = event.values[2] }
            Sensor.TYPE_GYROSCOPE -> { gyroX = event.values[0]; gyroY = event.values[1]; gyroZ = event.values[2] }
            Sensor.TYPE_ROTATION_VECTOR -> { rotX = event.values[0]; rotY = event.values[1]; rotZ = event.values[2] }
            Sensor.TYPE_MAGNETIC_FIELD -> { magX = event.values[0]; magY = event.values[1]; magZ = event.values[2] }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}

// --- Composable UI ---

@Composable
fun NumberPadScreen(
    modifier: Modifier = Modifier,
    onNumberClick: (String, () -> Unit) -> Unit,
    onOkClick: (String, () -> Unit) -> Unit // <-- CHANGED
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
            passcode += "*"
            passcodeVal += digit
            onNumberClick(digit) {
                // This reset block runs when the dialog is dismissed
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
            TextButton(
                label = "OK",
                onClick = {
                    onOkClick(passcodeVal) {
                        passcode = ""
                        passcodeVal = ""
                    }
                }
            )
            NumberButton(number = "0", onClick = handleDigitClick)

            // Backspace Button
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(80.dp)
                    .clickable {
                        if (passcode.isNotEmpty()) {
                            passcode = passcode.dropLast(1)
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
        NumberPadScreen(onNumberClick = { _, _ -> },onOkClick = { _ , _ -> })
    }
}

@Composable
fun TextButton(
    label: String,
    onClick: (String) -> Unit
) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(80.dp)
            .clip(CircleShape)
            // Use a different color or just keep it transparent/light to mimic common OK button styling
            .background(Color(0xFFE0E0E0)) // Slightly darker gray for action button
            .clickable { onClick(label) }
    ) {
        Text(
            text = label,
            color = Color.Black,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}


