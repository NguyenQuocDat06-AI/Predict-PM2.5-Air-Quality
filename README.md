# Đồ án Môn học: Toán Ứng Dụng
## Đề tài: Dự đoán nồng độ bụi mịn PM2.5 (Tập dữ liệu Beijing PM2.5 - Trạm Shunyi)

Dự án này tập trung vào việc nghiên cứu, cài đặt từ đầu bằng **NumPy/SciPy** (không sử dụng thư viện học máy bậc cao như `scikit-learn` cho việc tối ưu mô hình chính) và ứng dụng các thuật toán hồi quy tuyến tính/phi tuyến để dự đoán nồng độ bụi mịn PM2.5 dựa trên dữ liệu khí tượng và ô nhiễm thực tế tại trạm Shunyi, Bắc Kinh.

---

## Thông tin Nhóm 6
* **Lớp:** 24CTT2
* **Môn học:** Toán ứng dụng và thống kê- Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS)
* **Thành viên nhóm:**
  1. **24120282 - Nguyễn Quốc Đạt** (Trưởng nhóm)
  2. **24120308 - Mai Văn Hiển**
  3. **24120294 - Nguyễn Đức Duy**
  4. **24120184 - Lê Quốc Hưng**
  5. **24120155 - Trần Nam Việt**

---

## Cấu trúc Dự án
Dự án được phân chia thành 2 phần thực nghiệm chính và báo cáo khoa học chi tiết như sau:

```
Predict-PM2.5-Air-Quality/
├── part1/                             
│   ├── ols_implementation.py          
│   ├── ridge_lasso.py                
│   ├── cross_validation.py            
│   ├── residual_analysis.py          
│   ├── unit_tests.py                  
│   └── part1_notebook.ipynb           
│
├── part2/
│   ├── data/
│     ├── PRSA_Data_Shunyi_20130301-20170228.csv
│   ├── data_pipeline.py               
│   ├── advanced_methods.py           
│   ├── model_comparison.py
│   ├── part2_notebook.ipynb
│
├── report/                            
│   ├── report.tex               
│   ├── report.bib               
│   └── report.pdf               
│
├── requirements.txt                   
└── README.md
```

---

## Hướng dẫn Chạy các File Chương trình

### 1. Cài đặt thư viện cần thiết
Đảm bảo bạn đã cài đặt Python 3.10 trở lên. Cài đặt các thư viện phụ thuộc bằng lệnh sau:
```bash
pip install -r requirements.txt
```

### 2. Chạy Phần 1 (Mô phỏng & Kiểm thử thuật toán from scratch)
* **Chạy Unit Tests để xác minh tính chính xác của các hàm toán học tự viết:**
  ```bash
  python3 part1/unit_tests.py
  ```
  *(Khi chạy thành công, chương trình sẽ in thông báo: `Tất cả unit tests PASSED`)*

* **Khám phá và chạy mô phỏng Monte Carlo:**
  Khởi động Jupyter Notebook và mở file `part1/part1_notebook.ipynb` để chạy toàn bộ các cell phân tích mô phỏng:
  ```bash
  jupyter notebook part1/part1_notebook.ipynb
  ```

### 3. Chạy Phần 2 (Ứng dụng trên tập dữ liệu thực tế Beijing PM2.5)
* **Chạy so sánh mô hình tự động và xuất biểu đồ chẩn đoán phần dư:**
  Chạy trực tiếp script python để huấn luyện đồng thời 6 mô hình và xuất ra các biểu đồ chẩn đoán:
  ```bash
  python3 part2/model_comparison.py
  ```
  *(Script này sẽ in bảng so sánh hiệu năng MAE, RMSE, R² trực tiếp ra terminal)*

* **Chạy Jupyter Notebook để xem phân tích dữ liệu EDA và kết luận chi tiết:**
  Khởi động Jupyter Notebook và mở file `part2/part2_notebook.ipynb`:
  ```bash
  jupyter notebook part2/part2_notebook.ipynb
  ```

* **Biên dịch lại Báo cáo LaTeX từ đầu:**
  Di chuyển vào thư mục `report/` và sử dụng công cụ `pdflatex` để làm sạch và biên dịch báo cáo thành file PDF:
  ```bash
  cd report/
  pdflatex report_final.tex
  ```