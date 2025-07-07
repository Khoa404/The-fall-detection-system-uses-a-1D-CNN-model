import numpy as np
import pandas as pd

# Hàm tạo dữ liệu mô phỏng từ MPU6050 với các giai đoạn té ngã
def generate_mpu6050_data(num_samples, num_cycles):
    timestamps = np.arange(num_samples) / 50.0
    data = np.zeros((num_samples, 6))  # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    labels = np.zeros(num_samples)

    samples_per_cycle = num_samples // num_cycles  # 6,000 mẫu mỗi chu kỳ
    samples_per_phase_0 = samples_per_cycle // 6   # 1,000 mẫu mỗi giai đoạn của lớp 0 (3 giai đoạn)
    samples_per_impact = samples_per_cycle // 2    # 3,000 mẫu cho giai đoạn va chạm (lớp 1)

    for cycle in range(num_cycles):
        cycle_start = cycle * samples_per_cycle

        # 1. Bình thường (trước khi ngã)
        normal_start = cycle_start
        normal_end = normal_start + samples_per_phase_0
        data[normal_start:normal_end, 0] = np.random.uniform(-0.3, 0.3, samples_per_phase_0)
        data[normal_start:normal_end, 1] = np.random.uniform(-0.3, 0.3, samples_per_phase_0)
        data[normal_start:normal_end, 2] = np.random.uniform(0.8, 1.2, samples_per_phase_0)
        data[normal_start:normal_end, 3] = np.random.uniform(-50, 50, samples_per_phase_0)
        data[normal_start:normal_end, 4] = np.random.uniform(-50, 50, samples_per_phase_0)
        data[normal_start:normal_end, 5] = np.random.uniform(-50, 50, samples_per_phase_0)

        # 2. Trượt/mất cân bằng
        slip_start = normal_end
        slip_end = slip_start + samples_per_phase_0
        data[slip_start:slip_end, 0] = np.random.uniform(0.5, 1.0, samples_per_phase_0)
        data[slip_start:slip_end, 1] = np.random.uniform(0.3, 0.7, samples_per_phase_0)
        data[slip_start:slip_end, 2] = np.random.uniform(0.5, 0.8, samples_per_phase_0)
        data[slip_start:slip_end, 3] = np.random.uniform(-150, 150, samples_per_phase_0)
        data[slip_start:slip_end, 4] = np.random.uniform(-200, 200, samples_per_phase_0)
        data[slip_start:slip_end, 5] = np.random.uniform(-100, 100, samples_per_phase_0)

        # 3. Rơi tự do (tổng gia tốc < 0.7g, tốc độ góc > 200 độ/s)
        fall_start = slip_end
        fall_end = fall_start + samples_per_phase_0
        data[fall_start:fall_end, 0] = np.random.uniform(-0.3, 0.3, samples_per_phase_0)
        data[fall_start:fall_end, 1] = np.random.uniform(-0.3, 0.3, samples_per_phase_0)
        data[fall_start:fall_end, 2] = np.random.uniform(-0.2, 0.2, samples_per_phase_0)
        data[fall_start:fall_end, 3] = np.random.uniform(200, 300, samples_per_phase_0) * np.random.choice([-1, 1], samples_per_phase_0)
        data[fall_start:fall_end, 4] = np.random.uniform(200, 300, samples_per_phase_0) * np.random.choice([-1, 1], samples_per_phase_0)
        data[fall_start:fall_end, 5] = np.random.uniform(200, 300, samples_per_phase_0) * np.random.choice([-1, 1], samples_per_phase_0)

        # 4. Va chạm (tổng gia tốc > 2.0g, tốc độ góc < 50 độ/s)
        impact_start = fall_end
        impact_end = impact_start + samples_per_impact
        data[impact_start:impact_end, 0] = np.random.uniform(-1.5, 1.5, samples_per_impact)
        data[impact_start:impact_end, 1] = np.random.uniform(-1.0, 1.0, samples_per_impact)
        data[impact_start:impact_end, 2] = np.random.uniform(1.5, 2.0, samples_per_impact)
        data[impact_start:impact_end, 3] = np.random.uniform(-50, 50, samples_per_impact)
        data[impact_start:impact_end, 4] = np.random.uniform(-50, 50, samples_per_impact)
        data[impact_start:impact_end, 5] = np.random.uniform(-50, 50, samples_per_impact)

        # Gán nhãn: 0 cho 3 giai đoạn đầu, 1 cho giai đoạn va chạm
        labels[impact_start:impact_end] = 1

    # Thêm nhiễu Gaussian
    data[:, 0:3] += np.random.normal(0, 0.05, (num_samples, 3))  # accel
    data[:, 3:6] += np.random.normal(0, 5, (num_samples, 3))     # gyro

    # Giới hạn giá trị trong phạm vi thực tế của MPU6050
    data[:, 0:3] = np.clip(data[:, 0:3], -2.0, 2.0)  # accel
    data[:, 3:6] = np.clip(data[:, 3:6], -250, 250)  # gyro

    return np.column_stack((timestamps, data)), labels

# Tạo tập dữ liệu
num_samples = 30000
num_cycles = 5
data, labels = generate_mpu6050_data(num_samples, num_cycles)

# Kiểm tra tổng gia tốc và tổng tốc độ góc
accel_magnitude = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
gyro_magnitude = np.sqrt(data[:, 4]**2 + data[:, 5]**2 + data[:, 6]**2)

print(f"Tổng gia tốc nhỏ nhất: {np.min(accel_magnitude):.2f}g")
print(f"Tổng gia tốc lớn nhất: {np.max(accel_magnitude):.2f}g")
print(f"Số mẫu gia tốc trong khoảng [0.7g, 2.0g]: {np.sum((accel_magnitude >= 0.7) & (accel_magnitude <= 2.0))}")

print(f"Tổng tốc độ góc nhỏ nhất: {np.min(gyro_magnitude):.2f} độ/s")
print(f"Tổng tốc độ góc lớn nhất: {np.max(gyro_magnitude):.2f} độ/s")
print(f"Số mẫu tốc độ góc trong khoảng [50, 200] độ/s: {np.sum((gyro_magnitude >= 50) & (gyro_magnitude <= 200))}")

# Lưu dữ liệu vào file CSV
df = pd.DataFrame(data, columns=['timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
df['label'] = labels
df.to_csv('mpu6050_dataset.csv', index=False)
print("Đã tạo và lưu tập dữ liệu vào mpu6050_dataset.csv")