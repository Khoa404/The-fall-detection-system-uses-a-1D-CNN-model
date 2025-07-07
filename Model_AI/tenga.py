import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import tensorflow as tf

# Đọc dữ liệu
data = pd.read_csv('d:/AI/TeNga/mpu6050_dataset.csv')
required_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
if not all(col in data.columns for col in required_cols):
    raise ValueError("File CSV không chứa đủ các cột cần thiết!")
X = data[required_cols[:-1]].values
y = data['label'].values

# Chuẩn hóa dữ liệu (dữ liệu được chuẩn hóa theo z-score)
scaler = StandardScaler()      #scaler là một đối tượng của lớp StandardScaler
#khi gọi fit_transform, nó sẽ lưu trữ các tham số đã tính toán (trung bình và độ lệch chuẩn) để sử dụng sau này (ví dụ: để chuẩn hóa dữ liệu mới)
#scaler.mean_: Mảng chứa giá trị trung bình của từng cột trong X.
#scaler.scale_: Mảng chứa độ lệch chuẩn của từng cột trong X.
#scaler.var_: Mảng chứa phương sai (variance) của từng cột trong X (phương sai = bình phương của độ lệch chuẩn).
X = scaler.fit_transform(X)

print("Mean:", scaler.mean_)
print("Std:", scaler.scale_)

# Tạo cửa sổ thời gian
window_size = 5
X_windows = []           #được khởi tạo, ban đầu sẽ rỗng, Mỗi phần tử trong X_windows sẽ là một cửa sổ thời gian, chứa window_size mẫu dữ liệu liên tiếp từ X.
y_windows = []
for i in range(len(X) - window_size + 1):    #len(X): độ dài của X, vd X=7 thì số của sổ là 7-5+1 = 3 cửa sổ (từ 0 - 2) 3 lần lặp
    X_windows.append(X[i:i+window_size])     # khi i=0 thì x[0:5]   #với X[start: end] thì python sẽ lấy các giá trị từ start tới end -1 
    y_windows.append(y[i+window_size-1])     # khi i=0 thì y[0+5-1=4], y[4] là nhãn của phần tử x[4] là phần tử cuối cùng được lấy trong cửa sổ X ở trên
X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN
model = Sequential([
    Input(shape=(window_size, 6)),
    Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
    Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Biên dịch mô hình, Sử dụng hàm mất mát (loss function), bộ tối ưu (optimizer), và chỉ số đánh giá (metrics).
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()    # in tổng quan cấu trúc mô hình ra màn hình:
#•	Layer (type): tên lớp trong mạng.
#•	Output shape: kích thước đầu ra của mỗi lớp.
#•	Param #: số lượng tham số cần học trong lớp đó.
#•	Total params: tổng số tham số toàn mạng.
#•	Trainable params: tham số sẽ học trong quá trình training.


# Huấn luyện với EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
                                                     #validation_split=0.2 (tự động chia 20% tập train để làm tập theo dõi)
# batch_size = 32 : số mẫu / 1 batch
#ví dụ tập dữ liệu có 3200 mẫu thì sẽ chia ra được 3200/32 = 100 batch (lô)
#Với mỗi batch, mô hình sẽ thực hiện:
#Lan truyền xuôi (forward pass): dự đoán kết quả trên 32 mẫu.
#Tính hàm mất mát (loss).
#Lan truyền ngược (backpropagation): tính gradient.
#Cập nhật trọng số dựa trên gradient.
#Sau 100 batch, toàn bộ dữ liệu đã được dùng một lần ---→ đó là 1 epoch.

# Hàm tạo tập dữ liệu đại diện
def representative_dataset():
    indices = np.random.choice(X_train.shape[0], size=100, replace=False)
    # Giả sử X_train.shape = (1000, 5, 6). Shape[0] = 1000 //số cửa sổ
    #                                      Shape[1] = 5    //window_size
    #                                      Shape[2] = 6    //số đặc trưng
    #  chọn ngẫu nhiên 100 cửa sổ từ 1000 cửa sổ của tập X_train và không trùng lặp
    #
    for i in indices:
        yield [X_train[i:i+1].astype(np.float32)]

# Chuyển đổi sang TFLite với lượng tử hóa int8

# chuyển đổi các giá trị của mô hình (như trọng số và đầu vào/đầu ra) từ dạng số thực float32 sang int8
# Giảm kích thước của mô hình, tăng tốc độ tính toán, tiết kiệm năng lượng
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tạo một converter để chuyển đổi mô hình Keras (model) sang định dạng TensorFlow Lite (.tflite)
# tf.lite.TFLiteConverter: là trình chuyển đổi tích hợp của TensorFlow để biến mô hình TensorFlow (hoặc Keras) thành phiên bản nhẹ.
# .from_keras_model(model): là một phương thức dùng để khởi tạo converter từ mô hình Keras đã huấn luyện trước đó.

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# bật chế độ tối ưu hóa mặc định cho mô hình TensorFlow Lite khi chuyển đổi. (lượng tử hóa)

converter.representative_dataset = representative_dataset
# Tạo 1 tập dữ liệu đại diện vì TFLite cần biết phạm vi giá trị của dữ liệu đầu vào (input data) 
# để xác định các tham số lượng tử hóa (scale và zero point) cho từng tensor (trọng số, activations, đầu vào, đầu ra).

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# cho biết mô hình dùng kiểu toán tử nào  # tf.lite.OpsSet.TFLITE_BUILTINS_INT8 : chỉ dùng các toán tử hỗ trợ int8             # 1 số tầng ko hỗ trợ int8 như: softmax, Reshape, LayerNormalization, BatchNormalization,...

converter.inference_input_type = tf.int8
# Chỉ định kiểu dữ liệu đầu vào của mô hình TFLite là int8. 
# int8_value = round ((float_value/scale)+ zero_point)

converter.inference_output_type = tf.int8
# Chỉ định kiểu dữ liệu đầu ra của mô hình TFLite là int8.
# Chuyển int8 lại thành float32: float_value = (int8_value - zero_point) * scale

tflite_model = converter.convert()
# Thực hiện chuyển đổi mô hình Keras thành mô hình TFLite với các cấu hình đã thiết lập.
# Kết quả là tflite_model, một chuỗi bytes đại diện cho mô hình TFLite đã lượng tử hóa.

# Lưu mô hình
with open('fall_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Đã lưu mô hình TFLite với lượng tử hóa int8!")

# Kiểm tra mô hình TFLite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kiểm tra dự đoán trên 5 mẫu từ X_test
num_samples = 5
for i in range(min(num_samples, len(X_test))):
    test_sample = X_test[i:i+1]  # Dữ liệu đã chuẩn hóa từ scaler
    input_scale, input_zero_point = input_details[0]['quantization']
    test_sample_int8 = (test_sample / input_scale + input_zero_point).astype(np.int8)
    
    interpreter.set_tensor(input_details[0]['index'], test_sample_int8)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Chuyển đổi đầu ra int8 về giá trị xác suất (0-1)
    output_scale, output_zero_point = output_details[0]['quantization']
    output_float = (output.astype(np.float32) - output_zero_point) * output_scale
    
    print(f"Mẫu {i}:")
    print(f"  Dự đoán TFLite (int8): {output[0][0]}")
    print(f"  Dự đoán TFLite (float): {output_float[0][0]:.4f}")
    print(f"  Nhãn thực tế: {y_test[i]}")