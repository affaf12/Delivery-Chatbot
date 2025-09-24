# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

# --------------------------------------
# Helpers
# --------------------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lowcols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowcols:
            return lowcols[c.lower()]
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except Exception:
        return np.nan
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def safe_mean_series(s):
    return pd.to_numeric(s, errors="coerce").mean()

# --------------------------------------
# Core Q/A function (pandas-driven)
# --------------------------------------
def chat_with_delivery_data(question, df):
    q = question.lower().strip()

    # map common column names
    id_col = find_col(df, ["Delivery_person_ID", "delivery_person_id", "Delivery ID", "ID"])
    rating_col = find_col(df, ["Delivery_person_Ratings", "Delivery_person_Rating"])
    time_col = find_col(df, ["Time_taken (min)", "Time_taken_min", "Time_taken", "Time_taken_minutes"])
    city_col = find_col(df, ["City", "city", "Area"])
    multi_col = find_col(df, ["multiple_deliveries", "Multiple_deliveries"])
    vehicle_col = find_col(df, ["Type_of_vehicle", "vehicle_type"])
    order_type_col = find_col(df, ["Type_of_order", "type_of_order", "Order_Type"])
    time_order_col = find_col(df, ["Time_Orderd", "Time_Order_picked", "Order_Time"])
    weather_col = find_col(df, ["Weather_conditions", "Weather"])
    traffic_col = find_col(df, ["Road_traffic_density", "Traffic"])
    festival_col = find_col(df, ["Festival", "festival"])
    rest_lat = find_col(df, ["Restaurant_latitude"])
    rest_lon = find_col(df, ["Restaurant_longitude"])
    del_lat = find_col(df, ["Delivery_location_latitude"])
    del_lon = find_col(df, ["Delivery_location_longitude"])
    age_col = find_col(df, ["Delivery_person_Age", "Age"])
    vehicle_cond_col = find_col(df, ["Vehicle_condition", "vehicle_condition"])

    if time_col:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = pd.NA

    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # 1) Highest rating
    if "highest rating" in q or "highest rated" in q:
        if rating_col and id_col:
            best = df.loc[pd.to_numeric(df[rating_col], errors="coerce").idxmax()]
            return f"✅ Highest rating: {best.get(id_col,'N/A')} — rating {best.get(rating_col)}"
        return "I can't find rating or ID columns."

    # 2) Fastest on average
    if "fastest on average" in q:
        if id_col and not df["_time_numeric_"].isna().all():
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            if avg_times.empty:
                return "No valid numeric time data."
            fastest = avg_times.idxmin()
            return f"⚡ Fastest on average: {fastest} with {avg_times.min():.2f} minutes"
        return "Missing delivery ID or time column."

    # Fallback
    return "❓ I couldn't find an exact answer. Try rephrasing or asking about ratings, fastest deliveries, city averages, etc."

# --------------------------------------
# App start
# --------------------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")
st.markdown("Ask questions about your delivery dataset and get clean, data-driven answers.")

# Load dataset directly (no upload/URL)
DATA_FILE = "Zomato Dataset.csv"
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found! Please make sure 'Zomato Dataset.csv' is in the repo.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"❌ Could not read dataset: {e}")
    st.stop()

# Sidebar (just options, no upload/url)
col_left, col_right = st.columns([1, 3])

with col_left:
    st.subheader("📝 Options")

    with st.expander("📊 Delivery Performance"):
        st.markdown("- Which delivery person is the fastest on average?\n- What is the average delivery time per city?")
    with st.expander("👥 Customer & Order Insights"):
        st.markdown("- What types of orders take the longest?\n- Are deliveries slower during festivals?")

# Right panel: Only Answer + Question
with col_right:
    st.subheader("🤖 AI Response")
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = ""
    answer_box = st.text_area(" ", value=st.session_state.get("last_answer", ""), height=160)

    st.subheader("🔍 Enter your question")
    q_val = st.text_area("", value=st.session_state.get("question", ""), key="question_input", height=80)

    if st.button("🚀 Ask Question"):
        if not q_val.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            answer = chat_with_delivery_data(q_val, df)
            st.session_state["last_answer"] = answer
            st.session_state["question"] = q_val
