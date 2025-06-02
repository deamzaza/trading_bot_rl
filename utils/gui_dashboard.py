import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from utils.env_rl import CryptoTradingEnv
from utils.binance_utils import get_historical_data
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RL Trading Bot", layout="wide")

st.title("📈 AI Trading Bot Dashboard")
st.markdown("ระบบ Reinforcement Learning สำหรับเทรดคริปโตแบบอัตโนมัติ")

# --- Config ---
symbol = st.sidebar.selectbox("เลือกเหรียญ", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
start_date = st.sidebar.date_input("วันเริ่มต้น", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("วันสิ้นสุด", pd.to_datetime("2022-12-31"))

if st.sidebar.button("โหลดข้อมูล"):
    df = get_historical_data(symbol, start_date.strftime("%d %b, %Y"), end_date.strftime("%d %b, %Y"), "1h")
    st.session_state["df"] = df
    st.success("📊 โหลดข้อมูลเรียบร้อยแล้ว!")

# --- Display Chart ---
if "df" in st.session_state:
    st.subheader(f"ราคาย้อนหลังของ {symbol}")
    st.line_chart(st.session_state["df"]["close"])

    # Load model
    if os.path.exists(f"models/{symbol}_ppo_model.zip"):
        st.subheader("🤖 กำลังโหลดโมเดลเพื่อทดสอบ...")
        model = PPO.load(f"models/{symbol}_ppo_model.zip")

        env = CryptoTradingEnv(st.session_state["df"])
        obs = env.reset()
        rewards = []
        portfolio_values = []

        for _ in range(len(st.session_state["df"])):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            portfolio_values.append(env.net_worth)
            if done:
                break

        st.subheader("💰 Portfolio Value")
        st.line_chart(portfolio_values)

        st.subheader("📈 สรุปผลกำไร")
        st.write(f"📊 กำไรสุดท้าย: ${env.net_worth - env.initial_balance:,.2f}")

    else:
        st.warning("❗ ยังไม่มีโมเดลที่เทรนแล้วในโฟลเดอร์ `models/`")

else:
    st.info("กรุณากด 'โหลดข้อมูล' ก่อนเพื่อเริ่มระบบ")
