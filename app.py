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
st.title("Hybrid Prophet-LSTM Forecasting Engine")
st.markdown("""
Aplikasi ini menggabungkan kekuatan statistik **Facebook Prophet** (untuk tren), 
**LSTM Deep Learning** (untuk volatilitas), dan **Sentimen Pasar** guna memprediksi harga aset kripto.
**Data Source:** Binance Public API (Multi-Mirror).
""")

# --- SIDEBAR (INPUT USER) ---
st.sidebar.header("⚙️ Konfigurasi Model")

# Mapping nama koin ke Simbol Binance (Pair USDT)
coin_options = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Binance Coin (BNB)": "BNBUSDT",
    "Solana (SOL)": "SOLUSDT",
    "Ripple (XRP)": "XRPUSDT",
    "Cardano (ADA)": "ADAUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT"
}

selected_coin_name = st.sidebar.selectbox("Pilih Aset Kripto:", list(coin_options.keys()))
coin_symbol = coin_options[selected_coin_name]

days_history = st.sidebar.slider("Data Historis (Hari):", min_value=365, max_value=1000, value=1000, step=100)
test_days = st.sidebar.slider("Durasi Validasi (Hari):", min_value=30, max_value=365, value=90, step=30)
epochs = st.sidebar.slider("Training Epochs (LSTM):", min_value=5, max_value=50, value=20, step=5)

# --- FUNGSI 1: DATA LOADER (ANTI-BLOKIR / MULTI-MIRROR) ---
@st.cache_data(ttl=3600, show_spinner=False) 
def get_binance_data(symbol, limit):
    # Daftar URL alternatif jika URL utama diblokir ISP
    base_urls = [
        "https://api.binance.com",       # Utama
        "https://api1.binance.com",      # Mirror 1
        "https://api2.binance.com",      # Mirror 2
        "https://api3.binance.com",      # Mirror 3
        "https://data-api.binance.vision" # Vision (Seringkali aman)
    ]
    
    params = {'symbol': symbol, 'interval': '1d', 'limit': limit}
    
    for base_url in base_urls:
        url = f"{base_url}/api/v3/klines"
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'q_asset_vol', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
                ])
                df = df[['open_time', 'close']]
                df.columns = ['timestamp', 'y']
                df['ds'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
                df['y'] = df['y'].astype(float)
                df = df[['ds', 'y']].sort_values('ds')
                return df
        except Exception:
            continue
            
    st.error("Gagal terhubung ke semua server Binance. Cek koneksi internet Anda.")
    return None

# --- FUNGSI 2: SENTIMEN PASAR (FEAR & GREED) ---
@st.cache_data(ttl=3600)
def get_fear_greed():
    """Mengambil data Sentimen dari Alternative.me"""
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        return value, classification
    except Exception:
        return 50, "Neutral" # Fallback

# --- FUNGSI 3: BACKTESTING DENGAN STOP LOSS ---
def backtest_strategy(df_result, initial_capital=1000, stop_loss_pct=0.05):
    """
    Simulasi Trading Pintar:
    1. Beli jika Prediksi Naik > 1%
    2. Jual jika Prediksi Turun > 1% (Take Profit)
    3. [NEW] Cut Loss jika harga turun melebihi batas toleransi (Stop Loss)
    """
    cash = initial_capital
    position = 0 
    entry_price = 0
    portfolio_value = []
    fee = 0.001 
    
    for i in range(len(df_result) - 1):
        current_price = df_result['y'].iloc[i]
        predicted_next = df_result['Hybrid'].iloc[i+1]
        
        # Hitung Valuasi Aset Harian
        current_val = cash + (position * current_price)
        portfolio_value.append(current_val)

        # --- LOGIKA TRADING ---
        # 1. Cek Stop Loss (Safety First)
        if position > 0:
            loss_pct = (current_price - entry_price) / entry_price
            if loss_pct <= -stop_loss_pct:
                # Jual Paksa (Cut Loss)
                cash = position * current_price * (1 - fee)
                position = 0
                entry_price = 0
                continue # Lanjut ke hari esok

        # 2. Sinyal Beli
        if predicted_next > current_price * 1.01 and cash > 0:
            buy_amount = (cash * (1 - fee)) / current_price
            position = buy_amount
            entry_price = current_price
            cash = 0
            
        # 3. Sinyal Jual (Take Profit / Exit Normal)
        elif predicted_next < current_price * 0.99 and position > 0:
            cash = position * current_price * (1 - fee)
            position = 0
            entry_price = 0
            
    last_val = cash + (position * df_result['y'].iloc[-1])
    portfolio_value.append(last_val)
    df_result['Portfolio'] = portfolio_value
    return df_result, last_val

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
        split_idx = len(df) - test_days
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()

        # 1. Prophet
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
            
            progress_bar = st.progress(0)
            for i in range(epochs):
                self.lstm_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
                progress_bar.progress((i + 1) / epochs)
            progress_bar.empty()

        # 3. Predict
        with st.spinner('Sedang melakukan Forecasting...'):
            future_test = df_test[['ds']].copy()
            prophet_forecast = self.prophet_model.predict(future_test)

            last_residuals = df_train['residual'].values[-self.look_back:]
            curr_input = self.scaler.transform(last_residuals.reshape(-1, 1)).reshape(1, self.look_back, 1)
            
            lstm_preds = []
            for _ in range(len(df_test)):
                pred = self.lstm_model.predict(curr_input, verbose=0)
                lstm_preds.append(pred[0, 0])
                curr_input = np.append(curr_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
            
            lstm_correction = self.scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))

            result = df_test.copy()
            result['Prophet'] = prophet_forecast['yhat'].values
            result['Hybrid'] = result['Prophet'] + lstm_correction.flatten()
            
            return df_train, result

# --- MAIN EXECUTION ---
if st.button("Mulai Analisis"):
    # 1. Panggil Data
    df = get_binance_data(coin_symbol, days_history)
    
    if df is not None:
        st.success(f"Data {selected_coin_name} berhasil diambil! ({len(df)} baris)")
        
        # 2. Train Model
        model = HybridForecaster()
        df_train, df_result = model.train_predict(df, test_days, epochs)
        
        # 3. Ambil Sentimen
        fng_value, fng_label = get_fear_greed()
        
        # --- VISUALISASI SENTIMEN ---
        st.markdown("---")
        st.subheader("Market Psychology (Sentimen)")
        col_s1, col_s2 = st.columns([1, 3])
        col_s1.metric("Fear & Greed Index", f"{fng_value}/100", fng_label)
        col_s2.progress(fng_value / 100)
        
        # --- VISUALISASI FORECAST ---
        st.subheader("Visualisasi Hasil Forecasting")
        
        # Metrics Forecast
        mae_p = np.mean(np.abs(df_result['y'] - df_result['Prophet']))
        mae_h = np.mean(np.abs(df_result['y'] - df_result['Hybrid']))
        improvement = ((mae_p - mae_h) / mae_p) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Prophet Error (MAE)", f"${mae_p:.2f}")
        c2.metric("Hybrid Error (MAE)", f"${mae_h:.2f}")
        c3.metric("Improvement", f"{improvement:.2f}%", delta_color="normal")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['y'], mode='lines', name='Actual Price', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['Hybrid'], mode='lines', name='Hybrid Prediction', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=df_result['ds'], y=df_result['Prophet'], mode='lines', name='Prophet Baseline', line=dict(color='blue', width=1, dash='dot'), opacity=0.5))
        fig.update_layout(title=f"Validasi Model: {selected_coin_name}", xaxis_title="Tanggal", yaxis_title="Harga (USD)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- VISUALISASI BACKTESTING ---
        st.markdown("---")
        st.subheader("Simulasi Trading (Smart Backtesting)")
        
        # Slider Stop Loss
        stop_loss_input = st.slider("Atur Batas Toleransi Rugi (Stop Loss):", 1, 20, 5, help="Jika harga turun X% dari harga beli, robot akan otomatis jual.") / 100
        
        df_backtest, final_balance = backtest_strategy(df_result, initial_capital=1000, stop_loss_pct=stop_loss_input)
        roi = ((final_balance - 1000) / 1000) * 100
        
        c_p1, c_p2, c_p3 = st.columns(3)
        c_p1.metric("Modal Awal", "$1,000")
        c_p2.metric("Saldo Akhir", f"${final_balance:.2f}")
        roi_color = "normal" if roi >= 0 else "inverse"
        c_p3.metric("ROI (Profit/Loss)", f"{roi:.2f}%", delta_color=roi_color)
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df_backtest['ds'], y=df_backtest['Portfolio'], mode='lines', name='Hybrid Strategy (With Stop Loss)', line=dict(color='gold', width=3)))
        buy_hold_return = (df_result['y'] / df_result['y'].iloc[0]) * 1000
        fig_equity.add_trace(go.Scatter(x=df_result['ds'], y=buy_hold_return, mode='lines', name='Buy & Hold (Bitcoin)', line=dict(color='grey', dash='dot')))
        fig_equity.update_layout(title="Equity Curve: Algo vs HODL", xaxis_title="Tanggal", yaxis_title="Nilai Portofolio (USD)", template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_equity, use_container_width=True)

        # --- DATA TABLE ---
        with st.expander("Lihat Data Mentah"):
            st.dataframe(df_result)