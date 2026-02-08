import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math

class HybridForecaster:

    def __init__(self, seed: int = 42):
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = 60 # LSTM melihat 60 hari ke belakang (memperpanjang memori)
        
        # Container data
        self.df_train = None
        self.df_test = None
        self.test_forecast = None
        
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def generate_dummy_data(self, start_date='2021-01-01', days=1825):

        dates = pd.date_range(start=start_date, periods=days)
        
        # 1. Tren Jangka Panjang
        trend = np.linspace(100, 300, days)
        
        # 2. Musiman Tahunan & Mingguan
        yearly_season = 30 * np.sin(2 * np.pi * dates.dayofyear / 365)
        weekly_season = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
        
        # 3. Komponen Non-Linear Kompleks (Target LSTM)
        # Gelombang aneh yang berubah frekuensinya
        complex_wave = 20 * np.sin(dates.dayofyear / 20) * np.cos(dates.dayofyear / 100)
        
        noise = np.random.normal(0, 5, days)
        y = trend + yearly_season + weekly_season + complex_wave + noise
        
        print(f"[INFO] Data 5 Tahun ({days} hari) berhasil dibuat.")
        print(f"       Start: {dates[0].date()} | End: {dates[-1].date()}")
        return pd.DataFrame({'ds': dates, 'y': y})

    def _create_lstm_dataset(self, dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def train_and_evaluate(self, df, test_days=365):

        # 1. SPLIT DATA
        split_idx = len(df) - test_days
        self.df_train = df.iloc[:split_idx].copy()
        self.df_test = df.iloc[split_idx:].copy()
        
        print(f"\n[INFO] Pembagian Data:")
        print(f"       Data Latih: {len(self.df_train)} hari (Jan 2021 - Awal 2025)")
        print(f"       Data Uji  : {len(self.df_test)} hari (Awal 2025 - Feb 2026)")
        
        # --- PHASE 1: PROPHET (TRAINING) ---
        print("\n[STEP 1] Melatih Prophet pada Data Latih...")
        self.prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.prophet_model.fit(self.df_train)
        
        # Hitung Residual pada Data Latih
        future_train = self.prophet_model.make_future_dataframe(periods=0)
        forecast_train = self.prophet_model.predict(future_train)
        self.df_train['yhat_prophet'] = forecast_train['yhat'].values
        self.df_train['residual'] = self.df_train['y'] - self.df_train['yhat_prophet']
        
        # --- PHASE 2: LSTM (TRAINING) ---
        print("[STEP 2] Melatih LSTM pada Residual Data Latih...")
        
        # Scaling
        residuals = self.df_train['residual'].values.reshape(-1, 1)
        scaled_residuals = self.scaler.fit_transform(residuals)
        
        X_train, y_train = self._create_lstm_dataset(scaled_residuals, self.look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Model Architecture
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(64, return_sequences=False, input_shape=(self.look_back, 1)))
        self.lstm_model.add(Dense(32, activation='relu'))
        self.lstm_model.add(Dense(1))
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=5, verbose=0)
        self.lstm_model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1, callbacks=[es])
        
        # --- PHASE 3: EVALUATION (TESTING) ---
        print("\n[STEP 3] Melakukan Validasi pada Data Uji (1 Tahun Terakhir)...")
        self._evaluate_on_test_set()

    def _evaluate_on_test_set(self):
        """
        Memprediksi Data Uji dan membandingkan dengan Data Asli.
        """
        # A. Prediksi Prophet pada masa Data Uji
        future_test = self.df_test[['ds']].copy()
        prophet_forecast_test = self.prophet_model.predict(future_test)
        
        # B. Prediksi LSTM pada masa Data Uji (Recursive)
        # Ambil data residual terakhir dari Training sebagai 'start'
        last_residuals = self.df_train['residual'].values[-self.look_back:]
        last_residuals_scaled = self.scaler.transform(last_residuals.reshape(-1, 1))
        
        curr_input = last_residuals_scaled.reshape(1, self.look_back, 1)
        lstm_predictions_scaled = []
        
        steps = len(self.df_test) # Prediksi sepanjang data uji
        
        print(f"       -> Menggenerate {steps} hari prediksi LSTM...")
        
        for i in range(steps):
            pred = self.lstm_model.predict(curr_input, verbose=0)
            lstm_predictions_scaled.append(pred[0, 0])
            
            # Update window (Geser dan masukkan prediksi baru)
            new_observation = pred.reshape(1, 1, 1)
            curr_input = np.append(curr_input[:, 1:, :], new_observation, axis=1)
            
        # Kembalikan ke skala asli
        lstm_predictions = self.scaler.inverse_transform(np.array(lstm_predictions_scaled).reshape(-1, 1))
        
        # C. Gabungkan
        self.test_forecast = self.df_test.copy()
        self.test_forecast['yhat_prophet'] = prophet_forecast_test['yhat'].values
        self.test_forecast['lstm_correction'] = lstm_predictions
        self.test_forecast['hybrid_pred'] = self.test_forecast['yhat_prophet'] + self.test_forecast['lstm_correction']
        
        # D. Hitung Error (Metrik Akademis)
        mse = mean_squared_error(self.test_forecast['y'], self.test_forecast['hybrid_pred'])
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(self.test_forecast['y'], self.test_forecast['hybrid_pred'])
        
        print(f"\n[HASIL VALIDASI DATA UJI]")
        print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"MAE  (Mean Absolute Error)    : {mae:.4f}")
        print("Note: Semakin kecil error, semakin akurat model.")

    def visualize_validation(self):
        """Visualisasi Khusus: Train vs Test vs Prediksi"""
        plt.figure(figsize=(15, 7))
        
        # 1. Plot Data Latih (Hitam)
        plt.plot(self.df_train['ds'], self.df_train['y'], label='Data Latih (Training)', color='black', alpha=0.3)
        
        # 2. Plot Data Uji (Hijau - Data Asli Masa Depan)
        plt.plot(self.df_test['ds'], self.df_test['y'], label='Data Uji (Actual Test Data)', color='green')
        
        # 3. Plot Prediksi Hybrid (Merah Putus-putus)
        plt.plot(self.test_forecast['ds'], self.test_forecast['hybrid_pred'], 
                 label='Prediksi Hybrid (Validation)', color='red', linestyle='--', linewidth=2)
        
        plt.title('Evaluasi Model: Training (4 Tahun) vs Testing (1 Tahun Terakhir)', fontsize=14)
        plt.xlabel('Tahun')
        plt.ylabel('Nilai')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Zoom in ke bagian Test saja biar jelas
        plt.figure(figsize=(15, 5))
        plt.plot(self.df_test['ds'], self.df_test['y'], label='Data Uji (Actual)', color='green')
        plt.plot(self.test_forecast['ds'], self.test_forecast['hybrid_pred'], label='Prediksi Hybrid', color='red', linestyle='--')
        plt.title('Zoom-In: Perbandingan Data Asli vs Prediksi (2025-2026)', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    try:
        model = HybridForecaster(seed=42)
        
        # 1. Buat Data 5 Tahun (2021 s/d Feb 2026)
        df_all = model.generate_dummy_data(start_date='2021-01-01', days=1900)
        
        # 2. Latih & Evaluasi
        # Kita ambil 365 hari terakhir sebagai Data Uji (Test Set)
        model.train_and_evaluate(df_all, test_days=365)
        
        # 3. Visualisasi Hasil Uji
        model.visualize_validation()
        
        # Output DataFrame hasil uji (untuk dicek manual)
        print("\n--- Contoh Data Hasil Uji (5 Baris Terakhir Feb 2026) ---")
        print(model.test_forecast[['ds', 'y', 'yhat_prophet', 'hybrid_pred']].tail())

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()