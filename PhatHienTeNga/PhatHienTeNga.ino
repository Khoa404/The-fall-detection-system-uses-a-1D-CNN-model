#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_allocator.h>

#include <Wire.h>
#include <MPU6050_tockn.h>
#include <WiFi.h>
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"
#include <ESP_Mail_Client.h>

#include "model.h"

#include <FS.h>
#include <SPIFFS.h>
#include <NTPClient.h>
#include <WiFiUdp.h>

// Khởi tạo NTPClient để lấy thời gian
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 7 * 3600, 60000); // UTC+7, cập nhật mỗi 60s

// Định nghĩa tên file log
#define FALL_LOG_FILE "/fall_history.txt"

// Thông tin WiFi & MQTT
const char* ssid = " ";
const char* password = " ";
#define IO_USERNAME  " "
#define IO_KEY       " "
#define MQTT_SERVER  "io.adafruit.com"
#define MQTT_PORT    1883

// Thông tin SMTP (Gmail)
#define SMTP_SERVER "smtp.gmail.com"
#define SMTP_PORT 465
#define EMAIL_SENDER " "
#define EMAIL_PASSWORD " "
#define EMAIL_RECIPIENT " "

// Định nghĩa chân kết nối
#define BUZZER_PIN 15
#define BUTTON_PIN 4

WiFiClient client;
Adafruit_MQTT_Client mqtt(&client, MQTT_SERVER, MQTT_PORT, IO_USERNAME, IO_KEY);
Adafruit_MQTT_Publish fall_alert = Adafruit_MQTT_Publish(&mqtt, IO_USERNAME "/feeds/fall");
Adafruit_MQTT_Publish accel_data = Adafruit_MQTT_Publish(&mqtt, IO_USERNAME "/feeds/accel");
Adafruit_MQTT_Publish gyro_data = Adafruit_MQTT_Publish(&mqtt, IO_USERNAME "/feeds/gyro");

MPU6050 mpu(Wire);
SMTPSession smtp;

// Cấu hình TensorFlow Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(fall_detection_model_tflite);
static uint8_t tensor_arena[30 * 1024];
tflite::MicroInterpreter* interpreter;

bool daTeNga = false;
unsigned long lastReconnectTime = 0;
unsigned long lastMQTTUpdateTime = 0; // Thời gian gửi MQTT cuối cùng

// Cửa sổ thời gian (5 mẫu, 100ms/mẫu)
#define WINDOW_SIZE 5
float data_window[WINDOW_SIZE][6];  // Lưu 5 mẫu, mỗi mẫu có 6 đặc trưng
int window_index = 0;
bool window_full = false;

// Giá trị mean và std từ StandardScaler (6 đặc trưng)
float mean_values[6] = { 0.09139339,  0.06298399,  0.92505254,  0.5258811,  -0.56462044, -0.12921795};
float std_values[6] = { 0.46095246,  0.35008886,  0.46887669, 92.92475418, 96.97898587, 90.22106437};

// Giá trị scale và zero_point (cần lấy từ mô hình TFLite)
float input_scale = 0.024554690346121788;
int32_t input_zero_point = 14;
float output_scale = 0.00390625;
int32_t output_zero_point = -128;

// Hàm chuẩn hóa và lượng tử hóa dữ liệu sang int8
void normalize_and_quantize(float* input_data, int8_t* output_data, int size) {
    for (int i = 0; i < size; i++) {
        float normalized = (input_data[i] - mean_values[i % 6]) / std_values[i % 6];
        output_data[i] = (int8_t)(normalized / input_scale + input_zero_point);
    }
}

// Hàm kiểm tra bổ sung trên cửa sổ thời gian và tính toán giaToc, gyroMax
bool isPotentialFall(float &giaToc, float &gyroMax) {
    if (!window_full) return false;

    bool has_free_fall = false;
    bool has_impact = false;
    bool has_high_gyro = false;

    giaToc = 0.0;
    gyroMax = 0.0;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        float accelX = data_window[i][0];
        float accelY = data_window[i][1];
        float accelZ = data_window[i][2];
        float gyroX = data_window[i][3];
        float gyroY = data_window[i][4];
        float gyroZ = data_window[i][5];

        float tempGiaToc = sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ);
        float tempGyroMax = max(max(abs(gyroX), abs(gyroY)), abs(gyroZ));

        if (tempGiaToc > giaToc) giaToc = tempGiaToc;
        if (tempGyroMax > gyroMax) gyroMax = tempGyroMax;

        if (tempGiaToc < 0.6) has_free_fall = true;
        if (tempGiaToc > 1.8) has_impact = true;
        if (tempGyroMax > 150) has_high_gyro = true;
    }

    return has_free_fall && has_impact && has_high_gyro;
}

// Hàm kết nối WiFi
bool reconnectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return true;
    Serial.print(" Dang ket noi WiFi...");
    WiFi.begin(ssid, password);
    unsigned long start = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - start < 5000) {
        delay(500);
        Serial.print(".");
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("✅ Ket noi WiFi thanh cong!");
        return true;
    }
    Serial.println(" Ket noi WiFi that bai!");
    return false;
}

// Hàm kết nối MQTT
bool reconnectMQTT() {
    if (WiFi.status() != WL_CONNECTED) return false;
    if (mqtt.connected()) return true;
    Serial.print(" Dang ket noi MQTT...");
    for (int i = 0; i < 10; i++) {
        if (mqtt.connect()) {
            Serial.println(" Ket noi MQTT thanh cong!");
            return true;
        }
        delay(500);
    }
    Serial.println(" Ket noi MQTT that bai!");
    return false;
}

// Hàm gửi email
void sendEmail() {
    ESP_Mail_Session session;
    session.server.host_name = SMTP_SERVER;
    session.server.port = SMTP_PORT;
    session.login.email = EMAIL_SENDER;
    session.login.password = EMAIL_PASSWORD;

    SMTP_Message message;
    message.sender.name = "ESP32 Alert";
    message.sender.email = EMAIL_SENDER;
    message.priority = esp_mail_smtp_priority::esp_mail_smtp_priority_high;
    message.subject = "Canh Bao Te Nga";
    message.text.content = "Phat hien te nga! Vui long kiem tra ngay.";
    message.text.charSet = "utf-8";
    message.text.transfer_encoding = Content_Transfer_Encoding::enc_7bit;

    message.addRecipient("Recipient", EMAIL_RECIPIENT);

    if (!smtp.connect(&session)) {
        Serial.println(" Loi ket noi SMTP: " + String(smtp.errorReason()));
        return;
    }

    if (!MailClient.sendMail(&smtp, &message)) {
        Serial.println(" Loi gui email: " + String(smtp.errorReason()));
    } else {
        Serial.println(" Da gui email thanh cong!");
    }
    smtp.closeSession();
}

// Hàm chuyển đổi epoch time sang ngày/tháng/năm
void getDateFromEpoch(unsigned long epoch, int &day, int &month, int &year) {
    const int SECONDS_PER_DAY = 86400;
    const int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    unsigned long days = epoch / SECONDS_PER_DAY;
    epoch = epoch % SECONDS_PER_DAY;

    year = 1970;
    while (days >= 365) {
        bool isLeapYear = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        int daysInYear = isLeapYear ? 366 : 365;
        if (days >= daysInYear) {
            days -= daysInYear;
            year++;
        } else {
            break;
        }
    }

    month = 1;
    while (days > 0) {
        bool isLeapYear = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        int daysInThisMonth = daysInMonth[month - 1];
        if (month == 2 && isLeapYear) daysInThisMonth = 29;

        if (days >= daysInThisMonth) {
            days -= daysInThisMonth;
            month++;
        } else {
            break;
        }
    }

    day = days + 1; // +1 vì ngày bắt đầu từ 1
}

// Hàm ghi lịch sử té ngã vào SPIFFS
void logFall(float accelX, float accelY, float accelZ, float gyroX, float gyroY, float gyroZ) {
    timeClient.update(); // Đảm bảo thời gian được cập nhật
    int day, month, year;
    getDateFromEpoch(timeClient.getEpochTime(), day, month, year);
    String timestamp = timeClient.getFormattedTime() + " " + 
                      String(day) + "/" + 
                      String(month) + "/" + 
                      String(year);
    String logEntry = "Thoi gian: " + timestamp + 
                      ", AccelX: " + String(accelX, 3) + 
                      ", AccelY: " + String(accelY, 3) + 
                      ", AccelZ: " + String(accelZ, 3) + 
                      ", GyroX: " + String(gyroX, 3) + 
                      ", GyroY: " + String(gyroY, 3) + 
                      ", GyroZ: " + String(gyroZ, 3) + "\n";
    File file = SPIFFS.open(FALL_LOG_FILE, FILE_APPEND);
    if (!file) {
        Serial.println(" Loi mo file de ghi!");
        return;
    }
    if (file.print(logEntry)) {
        Serial.println(" Da ghi lich su te nga vao file: " + timestamp);
    } else {
        Serial.println(" Loi ghi file!");
    }
    file.close();
}

// Hàm đọc lịch sử té ngã từ SPIFFS
void readFallHistory() {
    File file = SPIFFS.open(FALL_LOG_FILE, FILE_READ);
    if (!file) {
        Serial.println(" Loi mo file de doc! Tao file moi.");
        file = SPIFFS.open(FALL_LOG_FILE, FILE_WRITE); // Tạo file mới nếu không tồn tại
        if (file) file.close();
        return;
    }
    Serial.println(" Noi dung lich su te nga:");
    while (file.available()) {
        Serial.write(file.read());
    }
    file.close();
}

// Hàm xóa lịch sử gần nhất
void deleteLatestFall() {
    File file = SPIFFS.open(FALL_LOG_FILE, FILE_READ);
    if (!file) {
        Serial.println(" Khong co lich su de xoa!");
        return;
    }

    // Đọc từng dòng vào một danh sách
    String lines[100]; // Giả sử tối đa 100 dòng
    int lineCount = 0;
    String currentLine = "";
    while (file.available()) {
        char c = file.read();
        if (c == '\n') {
            if (currentLine != "") { // Bỏ qua dòng trống
                lines[lineCount++] = currentLine;
            }
            currentLine = "";
        } else {
            currentLine += c;
        }
    }
    // Kiểm tra dòng cuối cùng nếu không có ký tự xuống dòng
    if (currentLine != "") {
        lines[lineCount++] = currentLine;
    }
    file.close();

    // Kiểm tra nếu không có dòng nào
    if (lineCount == 0) {
        SPIFFS.remove(FALL_LOG_FILE);
        Serial.println(" Khong co lich su de xoa!");
        return;
    }

    // Xóa dòng cuối cùng
    lineCount--;

    // Ghi lại các dòng còn lại
    file = SPIFFS.open(FALL_LOG_FILE, FILE_WRITE);
    if (!file) {
        Serial.println(" Khong the mo file de ghi lai!");
        return;
    }

    for (int i = 0; i < lineCount; i++) {
        file.print(lines[i]);
        file.print("\n"); // Thêm ký tự xuống dòng
    }
    file.close();

    Serial.println(" Da xoa lich su gan nhat!");
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    mpu.begin();
    mpu.calcGyroOffsets(true);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    digitalWrite(BUZZER_PIN, LOW);

    // Khởi tạo SPIFFS
    if (!SPIFFS.begin(true)) {
        Serial.println(" Loi khoi tao SPIFFS!");
        while (true);
    }
    Serial.println(" Khoi tao SPIFFS thanh cong!");

    // Kết nối WiFi và khởi tạo NTP
    if (!reconnectWiFi()) {
        Serial.println(" Khong ket noi duoc WiFi, thoat!");
        while (true);
    }
    timeClient.begin();
    timeClient.update();
    Serial.print(" Khoi tao NTPClient thanh cong, thoi gian: ");
    Serial.println(timeClient.getFormattedTime());

    readFallHistory(); // Đọc file lịch sử té ngã

    reconnectMQTT();

    // Khởi tạo TensorFlow Lite
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println(" Phien ban mo hinh khong tuong thich!");
        while (true);
    }
    // Sử dụng con trỏ đến micro_error_reporter
    tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(tensor_arena, sizeof(tensor_arena), &micro_error_reporter);
    if (allocator == nullptr) {
        Serial.println(" Loi tao MicroAllocator!");
        while (true);
    }
    interpreter = new tflite::MicroInterpreter(model, resolver, allocator, &micro_error_reporter);
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println(" Loi cap phat bo nho cho TensorFlow Lite!");
        while (true);
    }
    Serial.println(" Da khoi tao TensorFlow Lite thanh cong!");

    MailClient.networkReconnect(true);
}

void loop() {
    // Kiểm tra kết nối WiFi và MQTT mỗi 5 giây
    if (millis() - lastReconnectTime > 5000) {
        lastReconnectTime = millis();
        if (WiFi.status() != WL_CONNECTED) reconnectWiFi();
        if (!mqtt.connected()) reconnectMQTT();
    }

    mqtt.processPackets(50); // Xử lý các gói tin MQTT

    // Cập nhật dữ liệu từ cảm biến MPU6050
    mpu.update();
    float accelX = mpu.getAccX();
    float accelY = mpu.getAccY();
    float accelZ = mpu.getAccZ();
    float gyroX = mpu.getGyroX();
    float gyroY = mpu.getGyroY();
    float gyroZ = mpu.getGyroZ();

    // Lưu dữ liệu vào cửa sổ trượt
    data_window[window_index][0] = accelX;
    data_window[window_index][1] = accelY;
    data_window[window_index][2] = accelZ;
    data_window[window_index][3] = gyroX;
    data_window[window_index][4] = gyroY;
    data_window[window_index][5] = gyroZ;

    window_index = (window_index + 1) % WINDOW_SIZE;
    if (window_index == 0) window_full = true;

    // In dữ liệu cảm biến ra Serial Monitor
    Serial.print("X: "); Serial.print(accelX, 3); Serial.print(" g | ");
    Serial.print("Y: "); Serial.print(accelY, 3); Serial.print(" g | ");
    Serial.print("Z: "); Serial.print(accelZ, 3); Serial.println(" g");
    Serial.print("GyroX: "); Serial.print(gyroX);
    Serial.print(" | GyroY: "); Serial.print(gyroY);
    Serial.print(" | GyroZ: "); Serial.println(gyroZ);

    // Gửi giaToc và gyroMax lên MQTT mỗi 5 giây
    if (window_full && millis() - lastMQTTUpdateTime >= 5000) {
        float giaToc, gyroMax;
        isPotentialFall(giaToc, gyroMax); // Tính giaToc và gyroMax
        if (mqtt.connected()) {
            accel_data.publish(String(giaToc, 3).c_str());
            gyro_data.publish(String(gyroMax, 3).c_str());
            Serial.println("📤 Da gui giaToc: " + String(giaToc, 3) + ", gyroMax: " + String(gyroMax, 3) + " len MQTT");
        }
        lastMQTTUpdateTime = millis();
    }

    // Phát hiện té ngã khi cửa sổ đầy
    if (window_full) {
        float input_data[WINDOW_SIZE * 6];
        int idx = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            for (int j = 0; j < 6; j++) {
                input_data[idx++] = data_window[i][j];
            }
        }

        int8_t quantized_data[WINDOW_SIZE * 6];
        normalize_and_quantize(input_data, quantized_data, WINDOW_SIZE * 6);

        TfLiteTensor* input = interpreter->input(0);
        for (int i = 0; i < WINDOW_SIZE * 6; i++) {
            input->data.int8[i] = quantized_data[i];
        }

        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            Serial.println(" Loi chay mo hinh TensorFlow Lite!");
            return;
        }
        TfLiteTensor* output = interpreter->output(0);
        int8_t prediction_int8 = output->data.int8[0];
        float prediction = (prediction_int8 - output_zero_point) * output_scale;

        float giaToc, gyroMax;
        if (prediction > 0.8 && !daTeNga && isPotentialFall(giaToc, gyroMax)) {
            Serial.println(" CANH BAO: Phat hien te nga (AI)!");
            daTeNga = true;
            digitalWrite(BUZZER_PIN, HIGH);
            if (mqtt.connected()) {
                fall_alert.publish("1");
                accel_data.publish(String(giaToc, 3).c_str());
                gyro_data.publish(String(gyroMax, 3).c_str());
            }
            if (WiFi.status() == WL_CONNECTED) {
                sendEmail();
                logFall(accelX, accelY, accelZ, gyroX, gyroY, gyroZ);
            }
        }
    }

    // Tắt còi hoặc xóa lịch sử gần nhất tùy vào trạng thái
    if (digitalRead(BUTTON_PIN) == LOW) {
        if (daTeNga) {
            Serial.println(" Da nhan nut reset canh bao.");
            digitalWrite(BUZZER_PIN, LOW);
            daTeNga = false;
            if (mqtt.connected()) fall_alert.publish("0");
        } else {
            deleteLatestFall(); // Xóa lịch sử gần nhất
            readFallHistory();
        }
        delay(500); // Chống dội nút
    }

    delay(100); // 10Hz
}