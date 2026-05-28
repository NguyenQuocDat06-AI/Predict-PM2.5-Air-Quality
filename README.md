# Dự án dự đoán chất lượng không khí và PM2.5

Đây là dự án Machine Learning dự đoán nồng độ PM2.5 và chỉ số chất lượng không khí (AQI) dựa trên dữ liệu lịch sử và dự báo thời tiết.

## 🛠️ Cài đặt

### Yêu cầu
- Python 3.8+
- pip (Quản lý gói Python)

### Cài đặt Dependencies

1. Clone hoặc tải về repository:
```bash
git clone <repository-url>
cd Predict-PM2.5-Air-Quality
```

2. Tạo môi trường ảo (khuyến nghị):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 🚀 Hướng dẫn sử dụng

### 1. Khám phá dữ liệu
Chạy notebook để hiểu cấu trúc dữ liệu:
```bash
jupyter notebook 01_Data_Exploration.ipynb
```

### 2. Huấn luyện mô hình
Chạy notebook để huấn luyện mô hình:
```bash
jupyter notebook 02_Model_Training.ipynb
```

### 3. Đánh giá và so sánh mô hình
Chạy notebook để đánh giá hiệu năng các mô hình:
```bash
jupyter notebook 03_Model_Evaluation_Comparison.ipynb
```

### 4. Dự đoán chất lượng không khí (AQI)
Chạy notebook để sử dụng mô hình dự đoán AQI:
```bash
jupyter notebook 04_Air_Quality_Prediction.ipynb
```

## 📂 Cấu trúc dự án

```
Predict-PM2.5-Air-Quality/
├── data/                    # Dữ liệu đầu vào
├── notebooks/               # Các notebook Jupyter
│   ├── 01_Data_Exploration.ipynb  # Khám phá dữ liệu
│   ├── 02_Model_Training.ipynb      # Huấn luyện mô hình
│   ├── 03_Model_Evaluation_Comparison.ipynb  # Đánh giá mô hình
│   └── 04_Air_Quality_Prediction.ipynb        # Dự đoán AQI
├── models/                  # Mô hình đã huấn luyện
├── reports/                 # Báo cáo và kết quả
├── requirements.txt         # Dependencies
└── README.md                # Thông tin dự án
```

## 📝 Ghi chú

- Dự án này sử dụng tập dữ liệu AQI từ Cầu Giấy, Hà Nội.
- Các mô hình được sử dụng: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LSTM, và Transformer.
- Metric đánh giá: MAE, MSE, RMSE, R2-score.

## 🤝 Đóng góp

Đóng góp được hoan nghênh! Vui lòng tạo một branch riêng, thực hiện thay đổi, và gửi Pull Request.

## 📄 License

Dự án này được phát hành dưới MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Tác giả

Được tạo bởi **[Your Name]** - [email protected]