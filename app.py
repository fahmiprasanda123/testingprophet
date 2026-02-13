import streamlit as st
import pandas as pd
import numpy as np
import requests
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import plotly.graph_objects as go

# --- SETUP HALAMAN WEB ---
st.set_page_config(page_title="Crypto Hybrid Forecaster", layout="wide")

# --- JUDUL & DESKRIPSI ---
st.title("🚀 Hybrid Prophet-LSTM Forecasting Engine")
st.markdown("""
Aplikasi ini menggabungkan kekuatan statistik **Facebook Prophet** (untuk tren) 
dan **LSTM Deep Learning** (untuk volatilitas) guna memprediksi harga aset kripto.
**Data Source:** Binance Public API (Real-Time).
""")

# --- SIDEBAR (INPUT USER) ---
st.sidebar.header("⚙️ Konfigurasi Model")

# Mapping nama koin ke Simbol Binance (Pair USDT)
coin_options = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Ripple (XRP)": "XRPUSDT"
}

selected_coin_name = st.sidebar.selectbox("Pilih Aset Kripto:", list(coin_options.keys()))
coin_symbol = coin_options[selected_coin_name]

# Slider dibatasi max 1000 karena limit sekali call Binance API adalah 1000 candle
days_history = st.sidebar.slider("Data Historis (Hari):", min_value=365, max_value=1000, value=1000, step=100)
test_days = st.sidebar.slider("Durasi Validasi (Hari):", min_value=30, max_value=365, value=90, step=30)
epochs = st.sidebar.slider("Training Epochs (LSTM):", min_value=5, max_value=50, value=15, step=5)

# --- FUNGSI DATA LOADER (BINANCE API) ---
@st.cache_data(ttl=3600, show_spinner=False) 
def get_binance_data(symbol, limit):
    # Endpoint Binance untuk Data Candlestick (K-Lines)
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': '1d', # Data Harian
        'limit': limit    # Max 1000 data
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Format Binance: [Open Time, Open, High, Low, Close, Volume, ...]
            # Kita butuh index 0 (Time) dan index 4 (Close Price)
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'q_asset_vol', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
            ])
            
            # Ambil hanya kolom waktu dan harga close
            df = df[['open_time', 'close']]
            df.columns = ['timestamp', 'y']
            
            # Konversi tipe data
            df['ds'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
            df['y'] = df['y'].astype(float) # Harga di Binance string, harus jadi float
            
            # Bersihkan dan urutkan
            df = df[['ds', 'y']].sort_values('ds')
            return df
            
        else:
            st.error(f"Binance API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# --- CLASS MODEL ---
class HybridForecaster:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.look_back = 60
        self.prophet_model = None
        self.lstm_model = None
        tf.random.set_seed(42)
        np.random.seed(42)

    def _create_lstm_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def train_predict(self, df, test_days, epochs):
        # SPLIT
        split_idx = len(df) - test_days
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()

        # 1. PROPHET
        with st.spinner('Melatih Prophet (Baseline)...'):
            self.prophet_model = Prophet(daily_seasonality=True)
            self.prophet_model.fit(df_train)
            future_train = self.prophet_model.make_future_dataframe(periods=0)
            forecast_train = self.prophet_model.predict(future_train)
            df_train['residual'] = df_train['y'] - forecast_train['yhat'].values

        # 2. LSTM
        with st.spinner('Melatih LSTM (Residual Learning)...'):
            residuals = df_train['residual'].values.reshape(-1, 1)
            scaled_residuals = self.scaler.fit_transform(residuals)
            X_train, y_train = self._create_lstm_dataset(scaled_residuals)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            self.lstm_model = Sequential()
            self.lstm_model.add(LSTM(50, return_sequences=False, input_shape=(self.look_back, 1)))
            self.lstm_model.add(Dropout(0.2))
            self.lstm_model.add(Dense(1))
            self.lstm_model.compile(optimizer='adam', loss='mse')
            
            # Progress bar simulation
            progress_bar = st.progress(0)
            for i in range(epochs):
                self.lstm_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
                progress_bar.progress((i + 1) / epochs)
            progress_bar.empty()

        # 3. PREDICT
        with st.spinner('Sedang melakukan Forecasting...'):
            # Prophet Forecast
            future_test = df_test[['ds']].copy()
            prophet_forecast = self.prophet_model.predict(future_test)

            # LSTM Forecast
            last_residuals = df_train['residual'].values[-self.look_back:]
            curr_input = self.scaler.transform(last_residuals.reshape(-1, 1)).reshape(1, self.look_back, 1)
            
            lstm_preds = []
            for _ in range(len(df_test)):
                pred = self.lstm_model.predict(curr_input, verbose=0)
                lstm_preds.append(pred[0, 0])
                curr_input = np.append(curr_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            
            lstm_correction = self.scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))

            # Combine
            result = df_test.copy()
            result['Prophet'] = prophet_forecast['yhat'].values
            result['Hybrid'] = result['Prophet'] + lstm_correction.flatten()
            
            return df_train, result

# --- MAIN EXECUTION ---
if st.button("Mulai Analisis 🔮"):
    # 1. Panggil fungsi data (Binance Version)
    df = get_binance_data(coin_symbol, days_history)
    
    if df is not None:
        st.success(f"Data {selected_coin_name} berhasil diambil! ({len(df)} baris)")
        
        # 2. Inisialisasi Model
        model = HybridForecaster()
        
        # 3. Train & Predict
        df_train, df_result = model.train_predict(df, test_days, epochs)
        
        # --- METRICS ---
        mae_p = np.mean(np.abs(df_result['y'] - df_result['Prophet']))
        mae_h = np.mean(np.abs(df_result['y'] - df_result['Hybrid']))
        improvement = ((mae_p - mae_h) / mae_p) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Prophet Error (MAE)", f"${mae_p:.2f}")
        col2.metric("Hybrid Error (MAE)", f"${mae_h:.2f}")
        col3.metric("Improvement", f"{improvement:.2f}%", delta_color="normal")

        # --- VISUALISASI PLOTLY ---
        st.subheader("📊 Visualisasi Hasil Forecasting")
        
        fig = go.Figure()
        # Data Asli
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['y'], mode='lines', name='Actual Price', line=dict(color='green', width=2)))
        # Hybrid
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['Hybrid'], mode='lines', name='Hybrid Prediction', line=dict(color='red', dash='dash')))
        # Prophet
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['Prophet'], mode='lines', name='Prophet Baseline', line=dict(color='blue', width=1, dash='dot'), opacity=0.5))

        fig.update_layout(title=f"Validasi Model: {selected_coin_name}", xaxis_title="Tanggal", yaxis_title="Harga (USD)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- DATA TABLE ---
        with st.expander("Lihat Data Mentah"):
            st.dataframe(df_result)