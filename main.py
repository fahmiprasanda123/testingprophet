import pandas as pd
import numpy as np
import requests # Library untuk request API
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import math
import time

class HybridForecaster:

    def __init__(self, seed: int = 42):
        self.prophet_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = 60 
        
        self.df_train = None
        self.df_test = None
        self.test_forecast = None
        
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def get_crypto_data_api(self, coin_id='bitcoin', days='1800'):
        """
        [API INTEGRATION]
        """
        print(f"\n[INFO] Menghubungi CoinGecko API untuk data '{coin_id}'...")
        
        # URL Endpoint CoinGecko (Public API)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,       # Jumlah hari ke belakang (1800 hari ~= 5 tahun)
            'interval': 'daily' # Data harian
        }
        
        try:
            # Melakukan Request HTTP GET
            response = requests.get(url, params=params, timeout=10)
            
            # Cek jika sukses (Status 200)
            if response.status_code == 200:
                data = response.json()
                prices = data['prices'] # Format: [[timestamp, price], ...]
                
                # Convert ke Pandas DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'y'])
                
                # Convert timestamp (ms) ke datetime
                df['ds'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Hapus jam/menit/detik agar bersih (hanya tanggal)
                df['ds'] = df['ds'].dt.normalize()
                
                # Hapus duplikat tanggal (kadang API kasih data per jam di hari terakhir)
                df = df.drop_duplicates(subset=['ds'], keep='last')
                
                # Urutkan kolom
                df = df[['ds', 'y']]
                
                print(f"[SUKSES] Data {coin_id} berhasil ditarik via API!")
                print(f"         Total: {len(df)} baris data.")
                print(f"         Start: {df['ds'].iloc[0].date()} | End: {df['ds'].iloc[-1].date()}")
                return df
            
            else:
                raise ConnectionError(f"API Error {response.status_code}: {response.reason}")
                
        except Exception as e:
            print(f"[ERROR API] Gagal mengambil data: {e}")
            # Fallback darurat: Buat data dummy jika API mati total (agar codingan ga error)
            print("[WARNING] Menggunakan data dummy sebagai fallback agar program tetap jalan.")
            dates = pd.date_range(end=pd.Timestamp.now(), periods=int(days))
            y = np.linspace(10000, 50000, int(days)) + np.random.normal(0, 1000, int(days))
            return pd.DataFrame({'ds': dates, 'y': y})

    def _create_lstm_dataset(self, dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def train_and_evaluate(self, df, test_days=180):
        # 1. SPLIT DATA
        split_idx = len(df) - test_days
        self.df_train = df.iloc[:split_idx].copy()
        self.df_test = df.iloc[split_idx:].copy()
        
        print(f"\n[INFO] Skenario Split Data:")
        print(f"       Training: {len(self.df_train)} hari")
        print(f"       Testing : {len(self.df_test)} hari")
        
        # --- PHASE 1: PROPHET ---
        print("\n[STEP 1] Melatih Prophet...")
        self.prophet_model = Prophet(daily_seasonality=True) 
        self.prophet_model.fit(self.df_train)
        
        future_train = self.prophet_model.make_future_dataframe(periods=0)
        forecast_train = self.prophet_model.predict(future_train)
        self.df_train['yhat_prophet'] = forecast_train['yhat'].values
        self.df_train['residual'] = self.df_train['y'] - self.df_train['yhat_prophet']
        
        # --- PHASE 2: LSTM ---
        print("[STEP 2] Melatih LSTM...")
        
        residuals = self.df_train['residual'].values.reshape(-1, 1)
        scaled_residuals = self.scaler.fit_transform(residuals)
        
        X_train, y_train = self._create_lstm_dataset(scaled_residuals, self.look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, return_sequences=False, input_shape=(self.look_back, 1)))
        self.lstm_model.add(Dropout(0.2)) 
        self.lstm_model.add(Dense(1))
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        # Verbose 0 biar terminal ga penuh, ganti 1 kalo mau liat progress bar
        es = EarlyStopping(monitor='loss', patience=5, verbose=0) 
        self.lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[es])
        
        # --- PHASE 3: EVALUATION ---
        print("\n[STEP 3] Validasi Data...")
        self._evaluate_on_test_set()

    def _evaluate_on_test_set(self):
        # A. Prophet Baseline
        future_test = self.df_test[['ds']].copy()
        prophet_forecast_test = self.prophet_model.predict(future_test)
        
        # B. LSTM Prediction
        last_residuals = self.df_train['residual'].values[-self.look_back:]
        last_residuals_scaled = self.scaler.transform(last_residuals.reshape(-1, 1))
        
        curr_input = last_residuals_scaled.reshape(1, self.look_back, 1)
        lstm_predictions_scaled = []
        
        steps = len(self.df_test) 
        print(f"       -> Forecasting {steps} hari ke depan...")
        
        for i in range(steps):
            pred = self.lstm_model.predict(curr_input, verbose=0)
            lstm_predictions_scaled.append(pred[0, 0])
            new_observation = pred.reshape(1, 1, 1)
            curr_input = np.append(curr_input[:, 1:, :], new_observation, axis=1)
            
        lstm_predictions = self.scaler.inverse_transform(np.array(lstm_predictions_scaled).reshape(-1, 1))
        
        # C. Gabungkan
        self.test_forecast = self.df_test.copy()
        self.test_forecast['yhat_prophet'] = prophet_forecast_test['yhat'].values
        self.test_forecast['lstm_correction'] = lstm_predictions
        self.test_forecast['hybrid_pred'] = self.test_forecast['yhat_prophet'] + self.test_forecast['lstm_correction']
        
        # D. Metrics
        rmse_p = math.sqrt(mean_squared_error(self.test_forecast['y'], self.test_forecast['yhat_prophet']))
        rmse_h = math.sqrt(mean_squared_error(self.test_forecast['y'], self.test_forecast['hybrid_pred']))
        
        improv_rmse = ((rmse_p - rmse_h) / rmse_p) * 100
        
        print(f"\n{'='*55}")
        print(f"FINAL VERDICT: {ticker_name.upper()} MARKET DATA")
        print(f"{'='*55}")
        print(f"{'METRIK':<10} | {'PROPHET':<15} | {'HYBRID':<15} | {'DIFF':<10}")
        print(f"{'-'*55}")
        print(f"{'RMSE':<10} | {rmse_p:.4f}          | {rmse_h:.4f}          | {improv_rmse:.2f}%")
        print(f"{'='*55}")

    def visualize_validation(self, title_suffix=""):
        plt.figure(figsize=(15, 7))
        plt.plot(self.df_test['ds'], self.df_test['y'], label='Harga Asli (Actual)', color='green', linewidth=2)
        plt.plot(self.test_forecast['ds'], self.test_forecast['yhat_prophet'], 
                 label='Prophet (Baseline)', color='blue', linestyle=':', alpha=0.7)
        plt.plot(self.test_forecast['ds'], self.test_forecast['hybrid_pred'], 
                 label='Hybrid (Ours)', color='red', linestyle='--', linewidth=2)
        
        plt.title(f'Validasi Data Riil: {title_suffix}', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    try:
        model = HybridForecaster(seed=42)
        
        # Kita pakai ID 'bitcoin' untuk CoinGecko
        ticker_name = 'bitcoin' 
        
        # Request data 5 tahun terakhir (sekitar 1800 hari)
        # Ini MURNI REQUEST API, tidak ada file CSV
        df_market = model.get_crypto_data_api(coin_id=ticker_name, days='1800')
        
        # Latih & Evaluasi (Test pada 6 bulan terakhir)
        model.train_and_evaluate(df_market, test_days=180)
        
        model.visualize_validation(title_suffix=ticker_name.upper())

    except Exception as e:
        print(f"\n[ERROR FATAL]: {e}")
        import traceback
        traceback.print_exc()