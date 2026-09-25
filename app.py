import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# ============================================================
# पेज कॉन्फ़िगरेशन
# ============================================================
st.set_page_config(
    page_title="Gold Signal Scanner",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS स्टाइलिंग
# ============================================================
st.markdown("""
<style>
    .buy-signal {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white; padding: 20px; border-radius: 15px;
        text-align: center; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,200,83,0.4);
    }
    .sell-signal {
        background: linear-gradient(135deg, #d50000, #ff1744);
        color: white; padding: 20px; border-radius: 15px;
        text-align: center; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(213,0,0,0.4);
    }
    .neutral-signal {
        background: linear-gradient(135deg, #616161, #9e9e9e);
        color: white; padding: 20px; border-radius: 15px;
        text-align: center; font-size: 28px; font-weight: bold;
    }
    .metric-card {
        background: #1e1e2e; padding: 15px; border-radius: 10px;
        border: 1px solid #333; text-align: center;
    }
    .stMetric { background: #1e1e2e; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# डेटा लोडिंग फंक्शन
# ============================================================
@st.cache_data(ttl=300)  # हर 5 मिनट में रिफ्रेश
def load_gold_data(interval="1h", period="60d"):
    """Gold Futures (GC=F) डेटा लोड करें"""
    try:
        df = yf.download(
            "GC=F",
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=True
        )
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        st.error(f"डेटा लोडिंग एरर: {e}")
        return None

# ============================================================
# इंडिकेटर्स गणना
# ============================================================
def calculate_indicators(df):
    """सभी टेक्निकल इंडिकेटर्स कैलकुलेट करें"""
    df = df.copy()

    # RSI (14)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD (12, 26, 9)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']

    # बोलिंगर बैंड्स (20, 2)
    bb = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Middle'] = bb['BBM_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']

    # EMA (20, 50, 200)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)

    # स्टोचैस्टिक (14, 3, 3)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']

    # ADX (14)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']

    # ATR (14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # सुपरट्रेंड (10, 3)
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    if supertrend is not None:
        df['Supertrend'] = supertrend['SUPERT_10_3.0']
        df['Supertrend_Dir'] = supertrend['SUPERTd_10_3.0']

    # वॉल्यूम SMA (20)
    if df['Volume'].sum() > 0:
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()

    return df.dropna()

# ============================================================
# सिग्नल जनरेशन इंजन
# ============================================================
def generate_signals(df):
    """मल्टी-इंडिकेटर कन्फर्मेशन के साथ बाय/सेल सिग्नल जनरेट करें"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []

    # 1. RSI
    if latest['RSI'] < 30:
        buy_score += 2
        buy_reasons.append(f"RSI ओवरसोल्ड ({latest['RSI']:.1f})")
    elif latest['RSI'] > 70:
        sell_score += 2
        sell_reasons.append(f"RSI ओवरबॉट ({latest['RSI']:.1f})")

    # 2. MACD क्रॉसओवर
    if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
        buy_score += 2
        buy_reasons.append("MACD बुलिश क्रॉसओवर")
    elif prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
        sell_score += 2
        sell_reasons.append("MACD बेयरिश क्रॉसओवर")

    # 3. बोलिंगर बैंड्स
    if latest['Close'] <= latest['BB_Lower']:
        buy_score += 2
        buy_reasons.append("प्राइस लोअर बोलिंगर बैंड पर")
    elif latest['Close'] >= latest['BB_Upper']:
        sell_score += 2
        sell_reasons.append("प्राइस अपर बोलिंगर बैंड पर")

    # 4. EMA ट्रेंड
    if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']:
        buy_score += 1
        buy_reasons.append("EMA बुलिश अलाइनमेंट (20>50>200)")
    elif latest['EMA_20'] < latest['EMA_50'] < latest['EMA_200']:
        sell_score += 1
        sell_reasons.append("EMA बेयरिश अलाइनमेंट (20<50<200)")

    # 5. स्टोचैस्टिक
    if latest['Stoch_K'] < 20 and latest['Stoch_K'] > latest['Stoch_D']:
        buy_score += 1
        buy_reasons.append("स्टोचैस्टिक ओवरसोल्ड + बुलिश क्रॉस")
    elif latest['Stoch_K'] > 80 and latest['Stoch_K'] < latest['Stoch_D']:
        sell_score += 1
        sell_reasons.append("स्टोचैस्टिक ओवरबॉट + बेयरिश क्रॉस")

    # 6. ADX ट्रेंड स्ट्रेंथ
    if latest['ADX'] > 25:
        if latest['EMA_20'] > latest['EMA_50']:
            buy_score += 1
            buy_reasons.append(f"मजबूत ट्रेंड (ADX {latest['ADX']:.1f})")
        else:
            sell_score += 1
            sell_reasons.append(f"मजबूत ट्रेंड (ADX {latest['ADX']:.1f})")

    # 7. सुपरट्रेंड
    if 'Supertrend_Dir' in df.columns:
        if latest['Supertrend_Dir'] == 1 and prev['Supertrend_Dir'] == -1:
            buy_score += 2
            buy_reasons.append("सुपरट्रेंड बुलिश फ्लिप")
        elif latest['Supertrend_Dir'] == -1 and prev['Supertrend_Dir'] == 1:
            sell_score += 2
            sell_reasons.append("सुपरट्रेंड बेयरिश फ्लिप")

    # अंतिम सिग्नल
    max_score = max(buy_score, sell_score)
    total_possible = 11  # अधिकतम स्कोर

    if buy_score > sell_score and buy_score >= 5:
        signal = "BUY"
        confidence = min(int((buy_score / total_possible) * 100), 95)
        reasons = buy_reasons
    elif sell_score > buy_score and sell_score >= 5:
        signal = "SELL"
        confidence = min(int((sell_score / total_possible) * 100), 95)
        reasons = sell_reasons
    else:
        signal = "NEUTRAL"
        confidence = 0
        reasons = ["कोई मजबूत कन्फर्मेशन नहीं मिला"]

    # रिस्क लेवल्स (ATR आधारित)
    atr = latest['ATR']
    if signal == "BUY":
        entry = latest['Close']
        stop_loss = entry - (1.5 * atr)
        tp1 = entry + (1.5 * atr)
        tp2 = entry + (3.0 * atr)
        tp3 = entry + (4.5 * atr)
    elif signal == "SELL":
        entry = latest['Close']
        stop_loss = entry + (1.5 * atr)
        tp1 = entry - (1.5 * atr)
        tp2 = entry - (3.0 * atr)
        tp3 = entry - (4.5 * atr)
    else:
        entry = stop_loss = tp1 = tp2 = tp3 = latest['Close']

    return {
        'signal': signal,
        'confidence': confidence,
        'reasons': reasons,
        'entry': entry,
        'stop_loss': stop_loss,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'latest': latest
    }

# ============================================================
# चार्ट बनाएं
# ============================================================
def create_chart(df, signal_data, symbol="Gold Futures"):
    """कैंडलस्टिक चार्ट + इंडिकेटर्स"""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("प्राइस + EMA + बोलिंगर बैंड्स", "RSI", "MACD")
    )

    # कैंडलस्टिक
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="XAUUSD", increasing_line_color='#00c853',
        decreasing_line_color='#d50000'
    ), row=1, col=1)

    # EMA लाइन्स
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name="EMA 20",
                             line=dict(color='#ffeb3b', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name="EMA 50",
                             line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], name="EMA 200",
                             line=dict(color='#f44336', width=1.5)), row=1, col=1)

    # बोलिंगर बैंड्स
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper",
                             line=dict(color='rgba(100,181,246,0.5)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower",
                             line=dict(color='rgba(100,181,246,0.5)', width=1),
                             fill='tonexty', fillcolor='rgba(100,181,246,0.1)'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI",
                             line=dict(color='#ab47bc', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD",
                             line=dict(color='#29b6f6', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal",
                             line=dict(color='#ff7043', width=2)), row=3, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

# ============================================================
# मुख्य ऐप
# ============================================================
st.title("🥇 Gold (XAUUSD) Signal Scanner")
st.caption("मल्टी-इंडिकेटर कन्फर्मेशन आधारित बाय/सेल सिग्नल जनरेटर | World-Class Edition")

# साइडबार
with st.sidebar:
    st.header("⚙️ सेटिंग्स")

    timeframe = st.selectbox(
        "टाइमफ्रेम",
        ["15m", "1h", "4h", "1d"],
        index=1
    )

    interval_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    period_map = {"15m": "5d", "1h": "60d", "4h": "60d", "1d": "2y"}

    st.divider()
    st.markdown("**सिग्नल लॉजिक**")
    st.caption("RSI • MACD • BBands • EMA • Stoch • ADX • Supertrend")
    st.caption("कन्फर्मेशन स्कोर ≥ 5 होने पर ही सिग्नल")

    if st.button("🔄 डेटा रिफ्रेश करें", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# डेटा लोड करें
with st.spinner("गोल्ड डेटा लोड हो रहा है..."):
    df_raw = load_gold_data(
        interval=interval_map[timeframe],
        period=period_map[timeframe]
    )

if df_raw is None or len(df_raw) < 50:
    st.error("डेटा लोड नहीं हो पाया। कृपया रिफ्रेश करें या टाइमफ्रेम बदलें।")
    st.stop()

# इंडिकेटर्स कैलकुलेट करें
df = calculate_indicators(df_raw)

# सिग्नल जनरेट करें
signal_data = generate_signals(df)

# ============================================================
# सिग्नल डिस्प्ले
# ============================================================
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if signal_data['signal'] == "BUY":
        st.markdown(f'<div class="buy-signal">🟢 BUY</div>', unsafe_allow_html=True)
    elif signal_data['signal'] == "SELL":
        st.markdown(f'<div class="sell-signal">🔴 SELL</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="neutral-signal">⚪ NEUTRAL</div>', unsafe_allow_html=True)

with col2:
    st.metric("कॉन्फिडेंस", f"{signal_data['confidence']}%",
              delta=f"{len(signal_data['reasons'])} कन्फर्मेशन")

with col3:
    latest = signal_data['latest']
    st.metric("करंट प्राइस", f"${latest['Close']:.2f}",
              delta=f"{latest['Close'] - df['Close'].iloc[-2]:.2f}")

# सिग्नल कारण
st.markdown("### 📋 सिग्नल कन्फर्मेशन")
for reason in signal_data['reasons']:
    st.markdown(f"- ✅ {reason}")

# रिस्क लेवल्स
if signal_data['signal'] != "NEUTRAL":
    st.markdown("### 🎯 ट्रेड लेवल्स (ATR आधारित)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("एंट्री", f"${signal_data['entry']:.2f}")
    c2.metric("स्टॉप लॉस", f"${signal_data['stop_loss']:.2f}")
    c3.metric("TP 1", f"${signal_data['tp1']:.2f}")
    c4.metric("TP 2", f"${signal_data['tp2']:.2f}")
    c5.metric("TP 3", f"${signal_data['tp3']:.2f}")

# ============================================================
# चार्ट
# ============================================================
st.markdown("### 📈 टेक्निकल चार्ट")
fig = create_chart(df.tail(200), signal_data)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# इंडिकेटर टेबल
# ============================================================
with st.expander("📊 सभी इंडिकेटर वैल्यू देखें"):
    latest = signal_data['latest']
    indicator_df = pd.DataFrame({
        'इंडिकेटर': ['RSI', 'MACD', 'MACD Signal', 'BB Upper', 'BB Lower',
                     'EMA 20', 'EMA 50', 'EMA 200', 'Stoch K', 'Stoch D',
                     'ADX', 'ATR'],
        'वैल्यू': [
            f"{latest['RSI']:.2f}",
            f"{latest['MACD']:.2f}",
            f"{latest['MACD_Signal']:.2f}",
            f"{latest['BB_Upper']:.2f}",
            f"{latest['BB_Lower']:.2f}",
            f"{latest['EMA_20']:.2f}",
            f"{latest['EMA_50']:.2f}",
            f"{latest['EMA_200']:.2f}",
            f"{latest['Stoch_K']:.2f}",
            f"{latest['Stoch_D']:.2f}",
            f"{latest['ADX']:.2f}",
            f"{latest['ATR']:.2f}"
        ]
    })
    st.dataframe(indicator_df, use_container_width=True, hide_index=True)

st.caption("⚠️ अस्वीकरण: यह स्कैनर केवल शैक्षिक उद्देश्य के लिए है। कोई भी सिग्नल 100% सटीक नहीं होता। हमेशा अपने रिस्क मैनेजमेंट का पालन करें।")
