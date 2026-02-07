import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import itertools
from typing import Tuple, Optional, Dict

class BusinessForecaster:
  
    def __init__(self, seed: int = 42):

        self.model = None
        self.forecast_result = None
        self.best_params = None # Variable untuk menyimpan hasil tuning
        np.random.seed(seed)

    def generate_dummy_data(self, start_date: str = '2023-01-01', days: int = 365) -> pd.DataFrame:
        """
        Membuat data time series sintetis.
        """
        dates = pd.date_range(start=start_date, periods=days)
        
        # Komponen Tren & Musiman
        trend = np.linspace(100, 250, days)
        seasonality = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
        noise = np.random.normal(0, 10, days)
        
        y = trend + seasonality + noise
        
        df = pd.DataFrame({'ds': dates, 'y': y})
        print(f"[INFO] Data dummy berhasil dibuat: {days} baris data.")
        return df

    def optimize_model(self, df: pd.DataFrame) -> Dict:

        print("[INFO] Memulai proses Hyperparameter Tuning (ini mungkin memakan waktu)...")
        
        # 1. Tentukan Grid Parameter yang ingin diuji
        param_grid = {  
            'changepoint_prior_scale': [0.01, 0.1, 0.5], # Sensitivitas tren
            'seasonality_prior_scale': [0.1, 1.0, 10.0], # Kekuatan musiman
        }

        # Generate semua kombinasi parameter
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        
        rmses = []  # List untuk menyimpan nilai error

        # 2. Loop testing setiap parameter
        for params in all_params:
            # Train model sementara
            m = Prophet(**params, daily_seasonality=True, yearly_seasonality=True)
            m.fit(df)

            # Cross Validation: Potong data dan tes akurasi
            # initial: data latihan awal, horizon: seberapa jauh prediksi tes
            df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='30 days', parallel="processes")
            
            # Hitung error (RMSE)
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmse = df_p['rmse'].values[0]
            rmses.append(rmse)
            
            print(f"   > Test Params: {params} => Error (RMSE): {rmse:.4f}")

        # 3. Simpan parameter dengan error terkecil
        self.best_params = all_params[np.argmin(rmses)]
        print(f"[INFO] Tuning Selesai. Parameter Terbaik Ditemukan: {self.best_params}")
        return self.best_params

    def train_model(self, df: pd.DataFrame) -> None:

        print("[INFO] Memulai pelatihan model final...")
        
        if self.best_params:
            print(f"[INFO] Menggunakan parameter hasil tuning: {self.best_params}")
            # Unpack dictionary (**) ke dalam argumen Prophet
            self.model = Prophet(**self.best_params, daily_seasonality=True, yearly_seasonality=True)
        else:
            print("[INFO] Menggunakan parameter default (belum di-tuning).")
            self.model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        
        self.model.fit(df)
        print("[INFO] Model berhasil dilatih.")

    def make_prediction(self, periods: int = 60) -> pd.DataFrame:

        if not self.model:
            raise ValueError("Model belum dilatih. Jalankan train_model() terlebih dahulu.")

        print(f"[INFO] Melakukan prediksi untuk {periods} hari ke depan...")
        
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast_result = self.model.predict(future)
        return self.forecast_result

    def visualize_results(self, df_actual: pd.DataFrame) -> None:

        if self.forecast_result is None:
            raise ValueError("Belum ada hasil prediksi untuk di-plot.")

        print("[INFO] Membuat visualisasi...")

        # Plot 1: Prediksi vs Aktual
        fig1 = self.model.plot(self.forecast_result)
        plt.title('Prediksi Bisnis Professional (Tuned Model)', fontsize=16)
        plt.xlabel('Tanggal')
        plt.ylabel('Nilai')
        
        # Plot 2: Komponen
        fig2 = self.model.plot_components(self.forecast_result)
        
        plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        # 1. Instansiasi Kelas
        forecaster = BusinessForecaster(seed=123)

        # 2. Buat Data Dummy
        # 2 tahun data (730 hari)
        df_history = forecaster.generate_dummy_data(start_date='2022-01-01', days=730)

        # 3. Lakukan Tuning Parameter
        # looping hyperparameter untuk mendapatkan yang terbaik
        forecaster.optimize_model(df_history)

        # 4. Latih Model Final
        forecaster.train_model(df_history)

        # 5. Prediksi 90 Hari ke Depan
        forecast = forecaster.make_prediction(periods=90)

        # Tampilkan hasil
        print("\n--- Contoh Hasil Prediksi (5 Baris Terakhir) ---")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # 6. Visualisasi
        forecaster.visualize_results(df_history)

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")