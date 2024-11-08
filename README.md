# Anomaly-Detection

## Project Overview

**Anomaly-Detection** là một project tập trung vào phát hiện bất thường trong dữ liệu tiêu thụ năng lượng. Dự án sử dụng mô hình **Variational Autoencoder (VAE)** và **Long Short-Term Memory (LSTM)** để phát hiện các bất thường trong dữ liệu, sau đó sử dụng mô hình ngôn ngữ lớn (LLM) để giải thích nguyên nhân có thể gây ra những bất thường đó. Project này có tiềm năng hỗ trợ quản lý năng lượng hiệu quả hơn và cung cấp những hiểu biết sâu sắc về các yếu tố ảnh hưởng đến tiêu thụ năng lượng.

## Key Features

1. **Anomaly Detection**: Phát hiện bất thường trong dữ liệu tiêu thụ năng lượng bằng cách sử dụng mô hình VAE và LSTM.
2. **Explanation Generation**: Sử dụng LLM để giải thích các nguyên nhân có thể của các bất thường được phát hiện.
3. **Configurable**: Cấu hình các tham số qua file `params.yaml`, giúp tùy chỉnh dễ dàng.

## Installation

Cài đặt các thư viện này qua `requirements.txt`:
```bash
pip install -r requirements.txt
```

Chạy chương trình chính
```
python main.py
```

