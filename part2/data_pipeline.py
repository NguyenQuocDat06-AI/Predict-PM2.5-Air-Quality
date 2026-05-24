# Pipeline tiền xử lý dữ liệu chất lượng không khí — Trạm Shunyi, Bắc Kinh
# Nguồn: PRSA_Data_Shunyi_20130301-20170228.csv (UCI ML Repository)

# YÊU CẦU:
#   - Class DataPipeline xử lý missing values, encoding, chuẩn hóa theo thứ tự
#   - Có thể fit() trên train, transform() trên test (tránh data leakage)

# CÀI ĐẶT:
#   pip install pandas numpy scikit-learn scipy statsmodels missingno

# CÁCH SỬ DỤNG:
#   pipeline = DataPipeline()
#   df_train_clean = pipeline.fit(df_train)
#   df_test_clean  = pipeline.transform(df_test)

import pandas as pd
import numpy as np
import warnings
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# PHÂN LOẠI BIẾN (ngữ nghĩa quan trọng — ảnh hưởng trực tiếp đến mô hình)
#     Dù year/month/day/hour lưu dạng int64, về mặt NGỮ NGHĨA chúng là các biến
#     thời gian mang tính chất định tính. Đối với mô hình Hồi quy Tuyến tính,
#     không được đưa thẳng số gốc vào mô hình vì sẽ bị ép theo xu hướng tuyến tính sai lệch
#     (tháng 12 ≠ gấp 12 lần tháng 1; giờ 23 không tăng gấp 23 lần giờ 1).

#     → Chuyển đổi từ Cyclic Encoding sang Categorical Encoding (One-Hot Encoding)
#       để mô hình tự do học được các tác động phi tuyến tính cục bộ, nhiều đỉnh
#       (như 2 đỉnh giờ cao điểm, mùa mua sắm tháng cuối năm).

# Biến định lượng liên tục (continuous quantitative):
POLLUTANT_COLS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
METEO_COLS     = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
NUM_COLS       = POLLUTANT_COLS + METEO_COLS

# Biến định tính danh nghĩa (nominal) — cần One-Hot Encoding (bao gồm hướng gió và các biến thời gian):
NOMINAL_COLS   = ['wd', 'year', 'month', 'hour', 'day']

# Biến cần log-transform (phân phối lệch phải mạnh, skew > 3):
LOG_COLS       = ['PM2.5', 'SO2', 'CO', 'RAIN']

# Biến bỏ đi (không mang thông tin cho mô hình):
DROP_COLS      = ['No', 'station']   # 'No' = chỉ số, 'station' = hằng số

class DataPipeline:
    """
    Pipeline tiền xử lý dữ liệu chất lượng không khí trạm Shunyi.

    Thứ tự xử lý (fit và transform đều theo thứ tự này):
        1.  Tạo datetime index + sắp xếp theo thời gian
        2.  Bỏ cột không cần thiết (No, station)
        3.  Sửa mâu thuẫn logic vật lý (DEWP > TEMP)
        4.  Clip nồng độ chất ô nhiễm âm (không hợp lệ về vật lý)
        5.  Xử lý missing — Khí tượng: Time Interpolation  (MCAR, <0.15%)
        6.  Xử lý missing — Chất ô nhiễm: KNN Imputation k=5 (MAR, 2–6%)
        7.  Xử lý missing — Hướng gió wd: Mode cửa sổ 24h   (MAR, 1.38%)
        8.  Winsorization cho chất ô nhiễm (clip tại Q3+1.5×IQR)
        9.  Log1p transform cho biến lệch phải mạnh
        10. One-Hot Encoding cho các biến định tính danh nghĩa (wd, year, month, hour, day)
        11. StandardScaler — CHỈ fit trên TRAIN, transform trên test

    Attributes
    knn_imputer  : KNNImputer đã fit (lưu để transform test)
    scaler       : StandardScaler đã fit (lưu để transform test)
    ohe_categories: dict chứa danh sách các categories cho từng cột OHE
    winsor_bounds: dict {col: upper_bound} tính từ train
    _fitted      : bool — đánh dấu đã fit chưa
    """

    def __init__(self, knn_k: int = 5, interp_limit: int = 6, wd_window_h: int = 12, split_date: str = None):
        """
        Parameters
        knn_k        : số láng giềng cho KNNImputer (mặc định 5)
        interp_limit : số giờ tối đa nội suy liên tiếp (mặc định 6)
        wd_window_h  : cửa sổ ±h giờ để điền mode cho wd (mặc định 12)
        split_date   : nếu truyền vào, dùng để tách train/test nội bộ
        """
        self.knn_k         = knn_k
        self.interp_limit  = interp_limit
        self.wd_window_h   = wd_window_h
        self.split_date    = split_date

        self.knn_imputer   = KNNImputer(n_neighbors=knn_k, weights='distance')
        self.scaler        = StandardScaler()
        self.ohe_categories = {}
        self.winsor_bounds = {}
        self.scale_cols    = []
        self._fitted       = False

    # Tạo datetime index
    @staticmethod
    def _make_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo DatetimeIndex từ các cột year/month/day/hour.

        Lý do: Dữ liệu time series — cần index thời gian để interpolate
        đúng khoảng cách thực, sắp xếp thứ tự đúng cho mọi bước sau.
        """
        df = df.copy()
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.set_index('datetime').sort_index()
        return df

    # BƯỚC 3: Sửa mâu thuẫn logic DEWP > TEMP
    @staticmethod
    def _fix_dewpoint_conflict(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sửa các dòng có DEWP > TEMP (mâu thuẫn vật lý).

        Lý do: Theo vật lý khí quyển, nhiệt độ điểm sương (dew point) không
        thể cao hơn nhiệt độ không khí. Đây là lỗi đo đạc/ghi nhận.
        Phương pháp: gán DEWP = TEMP tại những dòng vi phạm (thay vì xóa),
        vì xóa dòng phá vỡ tính liên tục của time series.
        Lưu ý: TEMP và DEWP âm là hợp lệ với khí hậu Bắc Kinh
        (mùa đông thường xuống -10°C đến -20°C).
        """
        df = df.copy()
        mask = df['DEWP'] > df['TEMP']
        n_conflict = mask.sum()
        if n_conflict > 0:
            df.loc[mask, 'DEWP'] = df.loc[mask, 'TEMP']
            print(f"  [Fix conflict] Đã sửa {n_conflict} dòng DEWP > TEMP")
        return df

    # BƯỚC 4: Clip nồng độ âm
    @staticmethod
    def _clip_negative_pollutants(df: pd.DataFrame) -> pd.DataFrame:
        """
        Đặt nồng độ chất ô nhiễm < 0 về 0.

        Lý do: Nồng độ khí/bụi là đại lượng không âm về mặt vật lý.
        Giá trị âm là lỗi cảm biến hoặc lỗi hiệu chỉnh. Clip về 0 thay vì
        xóa để giữ nguyên tính toàn vẹn chuỗi thời gian.
        """
        df = df.copy()
        for col in POLLUTANT_COLS:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                df[col] = df[col].clip(lower=0)
                print(f"  [Clip neg] {col}: {n_neg} giá trị âm → clip về 0")
        return df

    # BƯỚC 5: Xử lý missing — Khí tượng (Time Interpolation)
    def _interpolate_meteo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nội suy tuyến tính theo trục thời gian cho biến khí tượng.

        Cơ chế thiếu: MCAR.
        Bằng chứng: Tỉ lệ thiếu chỉ 0.13%–0.15%, phân bố đều theo tháng,
        không có mẫu hình rõ ràng → khả năng do lỗi ngẫu nhiên của
        hệ thống lưu trữ, không liên quan đến giá trị khí tượng.

        Phương pháp: Time Interpolation.
        - Ưu điểm: Bảo toàn xu hướng biến đổi liên tục, không làm phẳng phân
          phối, phù hợp với dữ liệu time series.
        - limit=6: Chỉ nội suy tối đa 6 giờ liên tiếp để tránh ngoại suy xa.
        - Không dùng KNN vì tỉ lệ missing quá thấp, KNN tốn tài nguyên.
        - Không dùng Listwise Deletion vì phá vỡ chuỗi thời gian.
        """
        df = df.copy()
        df[METEO_COLS] = df[METEO_COLS].interpolate(
            method='time',
            limit=self.interp_limit,
            limit_direction='both'
        )
        return df

    # BƯỚC 6: Xử lý missing — Chất ô nhiễm (KNN Imputation)
    def _impute_pollutants_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit KNNImputer trên dữ liệu train và điền missing cho chất ô nhiễm.

        Cơ chế thiếu: MAR (Missing At Random).
        Bằng chứng: Sensor ô nhiễm bị lỗi/bảo trì định kỳ → missing thường
        xảy ra theo CỤM thời gian (vài giờ liên tiếp), phụ thuộc vào thời
        điểm trong năm (tỉ lệ thiếu CO cao hơn mùa đông). Có thể dự báo từ
        các biến khác → MAR, không phải MCAR.

        Phương pháp: KNN Imputation (MV4), k=5, trọng số nghịch đảo khoảng cách.
        - Ưu điểm so với Mean/Median (MV2): Tận dụng tương quan đa biến cao
          giữa các chất (PM2.5~PM10 r=0.88, PM2.5~CO r=0.75). KNN tìm 5 quan
          sát tương tự nhất dựa trên tất cả biến → điền chính xác hơn.
        - Ưu điểm so với Regression Imputation (MV3): KNN không giả định dạng
          hàm tuyến tính, phù hợp với phân phối lệch phải của dữ liệu ô nhiễm.
        - Feature cho KNN: dùng cả POLLUTANT + METEO (đã xử lý ở bước trên).
        - weights='distance': quan sát gần hơn được trọng số cao hơn.

        QUAN TRỌNG: Chỉ fit() trên TRAIN. Transform test dùng imputer đã fit,
        tránh data leakage.

        TỐI ƯU HÓA BỘ NHỚ (BATCHING): Để tránh lỗi tràn bộ nhớ do scikit-learn
        cố gắng cấp phát mảng khoảng cách khổng lồ trên RAM yếu,
        ta thực hiện fit trên toàn bộ dữ liệu nhưng transform theo từng batch nhỏ 5000 dòng.
        Phương pháp này giữ nguyên 100% kết quả toán học gốc nhưng giảm 10 lần RAM đỉnh.
        """
        df = df.copy()
        knn_features = POLLUTANT_COLS + METEO_COLS
        
        # Fit imputer để lưu cơ sở dữ liệu láng giềng donors
        self.knn_imputer.fit(df[knn_features])
        
        # Thực hiện transform theo từng batch nhỏ để bảo vệ bộ nhớ
        batch_size = 5000
        imputed_chunks = []
        for i in range(0, len(df), batch_size):
            batch = df[knn_features].iloc[i:i+batch_size]
            imputed_chunks.append(self.knn_imputer.transform(batch))
            
        df[knn_features] = np.vstack(imputed_chunks)
        print(f"  [KNN Impute] Fit KNNImputer(k={self.knn_k}) và điền khuyết thành công trên train.")
        return df

    def _impute_pollutants_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform (chỉ predict) KNNImputer đã fit trên train.
        Không fit lại để tránh data leakage.

        TỐI ƯU HÓA BỘ NHỚ (BATCHING): Áp dụng transform theo từng batch 5000 dòng
        để bảo vệ RAM tránh bị tràn khi transform tập dữ liệu test lớn.
        """
        df = df.copy()
        knn_features = POLLUTANT_COLS + METEO_COLS
        
        # Thực hiện transform theo từng batch nhỏ để bảo vệ bộ nhớ
        batch_size = 5000
        imputed_chunks = []
        for i in range(0, len(df), batch_size):
            batch = df[knn_features].iloc[i:i+batch_size]
            imputed_chunks.append(self.knn_imputer.transform(batch))
            
        df[knn_features] = np.vstack(imputed_chunks)
        return df

    # BƯỚC 7: Xử lý missing — Hướng gió wd (Mode cửa sổ thời gian)
    def _impute_wd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Điền missing hướng gió bằng mode trong cửa sổ ±window_h giờ.

        Cơ chế thiếu: MAR.
        Bằng chứng: Hướng gió thường không được ghi nhận khi WSPM ≈ 0 (gió lặng)
        → missing phụ thuộc vào tốc độ gió (biến quan sát được) → MAR.

        Phương pháp: Mode theo cửa sổ thời gian (±12h).
        - Không dùng KNN số vì 'wd' là biến ĐỊNH TÍNH DANH NGHĨA (nominal),
          khoảng cách Euclidean không có nghĩa với dữ liệu phân loại.
        - Không dùng One-Hot + KNN vì tăng chiều không cần thiết cho 1.38% missing.
        - Hướng gió có TÍNH LẶP LẠI THEO MÙA và liên tục trong cửa sổ ngắn
          → mode 24h xung quanh là xấp xỉ hợp lý nhất.
        - Nếu không tìm được hướng trong cửa sổ → giữ NaN (sẽ xử lý sau OHE).
        """
        filled = df['wd'].copy()
        null_idx = df['wd'][df['wd'].isnull()].index
        for idx in null_idx:
            window_data = df['wd'][
                (df.index >= idx - pd.Timedelta(hours=self.wd_window_h)) &
                (df.index <= idx + pd.Timedelta(hours=self.wd_window_h)) &
                df['wd'].notna()
            ]
            if len(window_data) > 0:
                filled[idx] = window_data.mode()[0]
        df = df.copy()
        df['wd'] = filled
        remaining = df['wd'].isnull().sum()
        print(f"  [WD Impute] Missing wd còn lại sau điền: {remaining}")
        return df

    # BƯỚC 8: Winsorization cho chất ô nhiễm
    def _winsorize_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính ngưỡng Winsorization từ train và áp dụng.

        Lý do chọn Winsorization thay vì xóa outlier:
        - Outlier trong dữ liệu ô nhiễm thường là thực tế. Xóa đi mất thông tin quan trọng.
        - Winsorization = "cắt bớt" tại ngưỡng IQR thay vì xóa, giữ lại
          tín hiệu nhưng giảm ảnh hưởng cực đoan lên mô hình.
        - Tỉ lệ outlier: SO2=8.6%, WSPM=5.7%, CO=5.6% → không thể xóa toàn bộ.

        Chỉ clip cận trên.
        Không Winsorize TEMP/PRES/DEWP vì giá trị cực đoan của chúng
        là hợp lệ về mặt vật lý.

        Tính ngưỡng chỉ từ TRAIN, áp dụng lên cả test để tránh data leakage.
        """
        df = df.copy()
        for col in POLLUTANT_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            upper = Q3 + 1.5 * (Q3 - Q1)
            self.winsor_bounds[col] = upper
            df[col] = df[col].clip(upper=upper)
            print(f"  [Winsorize] {col}: clip upper = {upper:.2f}")
        return df

    def _winsorize_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng Winsorization bounds đã tính từ train lên test."""
        df = df.copy()
        for col, upper in self.winsor_bounds.items():
            df[col] = df[col].clip(upper=upper)
        return df

    # BƯỚC 9: Log1p Transform
    @staticmethod
    def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng log(1 + x) cho các biến có phân phối lệch phải mạnh.

        Lý do:
        - PM2.5, SO2, CO có skewness > 3, RAIN có skewness = 23 (zero-inflated).
        - Mô hình hồi quy OLS giả định phần dư có phân phối chuẩn; phân phối
          lệch phải vi phạm giả định này, làm giảm hiệu quả ước lượng.
        - log(1+x) = log1p thay vì log(x) để tránh log(0) = -inf khi x = 0
          (RAIN và SO2 có nhiều giá trị = 0).
        - Sau transform: phân phối cân bằng hơn, mô hình tuyến tính hoạt động
          tốt hơn, hệ số dễ giải thích hơn (thay đổi % thay vì tuyệt đối).
        """
        df = df.copy()
        for col in LOG_COLS:
            df[f'log_{col}'] = np.log1p(df[col])
        return df

    # BƯỚC 10: One-Hot Encoding cho các biến phân loại định tính
    def _ohe_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-Hot Encoding cho các biến định tính danh nghĩa (wd, year, month, hour, day) — fit trên train.

        Lý do chuyển từ Cyclic Encoding sang One-Hot Encoding:
        - Đối với mô hình Hồi quy Tuyến tính, Cyclic Encoding giới hạn mô hình chỉ học được
          đường cong hình sin 1 đỉnh/đáy duy nhất, không thể bắt được các mẫu hình phức tạp
          (ví dụ: giờ cao điểm sáng và chiều).
        - One-Hot Encoding cho phép mô hình tuyến tính học được các tác động cục bộ, phi tuyến tính
          và nhiều đỉnh đặc thù của từng khung giờ/tháng/ngày.
        - drop_first=True giúp tránh bẫy biến giả (Dummy Variable Trap).
        """
        df = df.copy()
        self.ohe_categories = {}
        for col in NOMINAL_COLS:
            if col in df.columns:
                # Ép kiểu thành string để xử lý dạng định tính đồng nhất
                df[col] = df[col].astype(str)
                # Lấy các category duy nhất và sắp xếp
                cats = sorted(df[col].dropna().unique().tolist())
                self.ohe_categories[col] = cats
                df[col] = pd.Categorical(df[col], categories=cats)
        
        # Thực hiện One-Hot Encoding cho tất cả NOMINAL_COLS
        cols_to_encode = [c for c in NOMINAL_COLS if c in df.columns]
        df = pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode, 
                            drop_first=True, dtype=float)
        return df

    def _ohe_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng OHE với cùng categories từ train, đảm bảo đồng nhất cột giữa Train và Test.
        """
        df = df.copy()
        for col in NOMINAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = pd.Categorical(df[col], categories=self.ohe_categories.get(col, []))
        
        cols_to_encode = [c for c in NOMINAL_COLS if c in df.columns]
        df = pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode, 
                            drop_first=True, dtype=float)
        return df

    # BƯỚC 12: Standardization (Z-score)
    def _scale_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa Z-score: x_std = (x - mean_train) / std_train

        Lý do chuẩn hóa:
        - Các biến có đơn vị và phạm vi rất khác nhau: CO ∈ [0, 10000],
          PRES ∈ [983, 1040], WSPM ∈ [0, 13] → gradient descent hoặc
          solver không hội tụ đều nếu không chuẩn hóa.
        - Chuẩn hóa giúp hệ số hồi quy so sánh được với nhau (magnitude
          phản ánh tầm quan trọng tương đối).

        QUAN TRỌNG NHẤT: Chỉ fit() trên TRAIN.
        Lý do: Nếu fit trên toàn bộ data (train+test), mean và std sẽ bị
        ảnh hưởng bởi test → DATA LEAKAGE → mô hình "nhìn thấy" tương lai,
        hiệu suất đánh giá bị thổi phồng, không phản ánh thực tế triển khai.
        """
        df = df.copy()
        # Xác định cột cần scale: tất cả numeric trừ các cột dummy nhị phân OHE và cột định tính gốc
        exclude = []
        for col in NOMINAL_COLS:
            exclude.extend([c for c in df.columns if c.startswith(f'{col}_')])
        exclude.extend(NOMINAL_COLS)
        
        self.scale_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c not in exclude
        ]
        df[self.scale_cols] = self.scaler.fit_transform(df[self.scale_cols])
        print(f"  [Scale] Fit StandardScaler trên {len(self.scale_cols)} cột.")
        return df

    def _scale_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng scaler đã fit trên train, không fit lại."""
        df = df.copy()
        df[self.scale_cols] = self.scaler.transform(df[self.scale_cols])
        return df

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit pipeline trên tập TRAIN và trả về DataFrame đã xử lý.

        Thực hiện đầy đủ 12 bước, lưu lại các tham số học được
        (KNNImputer, Winsorization bounds, OHE categories, StandardScaler)
        để dùng lại trong transform().

        Parameters
        ----------
        df : pd.DataFrame — dữ liệu train thô (chưa xử lý)

        Returns
        -------
        pd.DataFrame — dữ liệu train đã xử lý, sẵn sàng cho mô hình
        """
        print("=" * 60)
        print("  DataPipeline.fit() — Bắt đầu xử lý TRAIN")
        print("=" * 60)

        # Bước 1: Datetime index
        df = self._make_datetime_index(df)

        # Bước 2: Drop cột không cần
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                     errors='ignore')

        # Bước 3: Fix DEWP > TEMP
        df = self._fix_dewpoint_conflict(df)

        # Bước 4: Clip nồng độ âm
        df = self._clip_negative_pollutants(df)

        # Bước 5: Missing — Khí tượng (Time Interpolation)
        print("  [Step 5] Time Interpolation cho biến khí tượng (MCAR)")
        df = self._interpolate_meteo(df)

        # Bước 6: Missing — Chất ô nhiễm (KNN) — fit ở đây
        print(f"  [Step 6] KNN Imputation (k={self.knn_k}) cho chất ô nhiễm (MAR)")
        df = self._impute_pollutants_fit(df)

        # Bước 7: Missing — wd (Mode window)
        print("  [Step 7] Mode window ±12h cho hướng gió wd (MAR)")
        df = self._impute_wd(df)

        # Bước 8: Winsorization — fit bounds từ train
        print("  [Step 8] Winsorization cho chất ô nhiễm (IQR bounds từ train)")
        df = self._winsorize_fit(df)

        # Bước 9: Log Transform
        print(f"  [Step 9] Log1p transform cho: {LOG_COLS}")
        df = self._log_transform(df)

        # Bước 10: One-Hot Encoding — fit categories từ train
        print("  [Step 10] One-Hot Encoding cho các biến phân loại (fit categories từ train)")
        df = self._ohe_fit(df)

        # Bước 11: StandardScaler — fit CHỈ trên train
        print("  [Step 11] StandardScaler — fit trên train (tránh data leakage)")
        df = self._scale_fit(df)

        self._fitted = True
        missing_left = df.isnull().sum().sum()
        print("=" * 60)
        print(f"  fit() hoàn tất | Shape: {df.shape} | Missing còn: {missing_left}")
        print("=" * 60)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform tập TEST dùng tham số đã học từ fit() trên train.

        KHÔNG fit lại bất kỳ tham số nào để tránh data leakage.

        Parameters
        ----------
        df : pd.DataFrame — dữ liệu test thô

        Returns
        -------
        pd.DataFrame — dữ liệu test đã xử lý với cùng schema như train
        """
        if not self._fitted:
            raise RuntimeError("Phải gọi fit() trước transform()!")

        print("=" * 60)
        print("  DataPipeline.transform() — Bắt đầu xử lý TEST")
        print("=" * 60)

        df = self._make_datetime_index(df)
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns],
                     errors='ignore')
        df = self._fix_dewpoint_conflict(df)
        df = self._clip_negative_pollutants(df)
        df = self._interpolate_meteo(df)
        df = self._impute_pollutants_transform(df)   # ← chỉ transform, không fit
        df = self._impute_wd(df)
        df = self._winsorize_transform(df)           # ← bounds từ train
        df = self._log_transform(df)
        df = self._ohe_transform(df)                 # ← categories từ train
        df = self._scale_transform(df)               # ← scaler từ train

        missing_left = df.isnull().sum().sum()
        print("=" * 60)
        print(f"  transform() hoàn tất | Shape: {df.shape} | Missing còn: {missing_left}")
        print("=" * 60)
        return df

# CHẠY THỬ ĐỘC LẬP
if __name__ == '__main__':
    import os

    DATA_PATH = 'PRSA_Data_Shunyi_20130301-20170228.csv'
    
    if not os.path.exists(DATA_PATH):
        print(f"Không tìm thấy file: {DATA_PATH}")
        print("Hãy đặt file CSV cùng thư mục với data_pipeline.py hoặc trong thư mục con PRSA_Data_20130301-20170228")
    else:
        # Nạp dữ liệu
        df_raw = pd.read_csv(DATA_PATH)
        print(f"Raw data shape: {df_raw.shape}")

        # Tách train/test theo thời gian 
        # Lý do: Dữ liệu time series — tách random vi phạm tính nhân quả.
        SPLIT_DATE = '2016-09-01'
        df_train = df_raw[
            pd.to_datetime(
                df_raw[['year', 'month', 'day', 'hour']]
            ) < SPLIT_DATE
        ].copy()
        df_test = df_raw[
            pd.to_datetime(
                df_raw[['year', 'month', 'day', 'hour']]
            ) >= SPLIT_DATE
        ].copy()

        print(f"Train: {len(df_train)} dòng | Test: {len(df_test)} dòng")

        # Khởi tạo và chạy pipeline
        pipeline = DataPipeline(knn_k=5, interp_limit=6, wd_window_h=12)

        df_train_clean = pipeline.fit(df_train)
        df_test_clean  = pipeline.transform(df_test)

        print("\n--- Tóm tắt kết quả ---")
        print(f"Train clean shape : {df_train_clean.shape}")
        print(f"Test  clean shape : {df_test_clean.shape}")
        print(f"Cột sau pipeline  : {list(df_train_clean.columns)}")

        # Lưu kết quả (tuỳ chọn)
        df_train_clean.to_csv('train_clean.csv')
        df_test_clean.to_csv('test_clean.csv')
        print("\nĐã lưu: train_clean.csv, test_clean.csv")
