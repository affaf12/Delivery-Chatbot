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

# --------------------------------------
# Core Q/A function
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
    weather_col = find_col(df, ["Weather_conditions", "Weather"])
    traffic_col = find_col(df, ["Road_traffic_density", "Traffic"])
    festival_col = find_col(df, ["Festival", "festival"])
    rest_lat = find_col(df, ["Restaurant_latitude"])
    rest_lon = find_col(df, ["Restaurant_longitude"])
    del_lat = find_col(df, ["Delivery_location_latitude"])
    del_lon = find_col(df, ["Delivery_location_longitude"])
    age_col = find_col(df, ["Delivery_person_Age", "Age"])
    vehicle_cond_col = find_col(df, ["Vehicle_condition", "vehicle_condition"])

    # prepare time column
    if time_col:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = pd.NA

    # distance
    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # -------- Rules --------
    if "highest rating" in q:
        if rating_col and id_col:
            best = df.loc[pd.to_numeric(df[rating_col], errors="coerce").idxmax()]
            return f"✅ Highest rating: {best.get(id_col)} — {best.get(rating_col)}"

    if "fastest" in q:
        if id_col:
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            fastest = avg_times.idxmin()
            return f"⚡ Fastest: {fastest} — {avg_times.min():.2f} min"

    if "average" in q and "city" in q:
        if city_col:
            avg = df.groupby(city_col)["_time_numeric_"].mean()
            return avg.to_string()

    if "multiple" in q:
        if multi_col:
            avg = df.groupby(multi_col)["_time_numeric_"].mean()
            return f"📦 Multiple deliveries impact:\n{avg.to_string()}"

    if "vehicle" in q:
        if vehicle_col:
            avg = df.groupby(vehicle_col)["_time_numeric_"].mean()
            return f"🚲 Vehicle efficiency:\n{avg.to_string()}"

    if "order type" in q or "types of orders" in q:
        if order_type_col:
            avg = df.groupby(order_type_col)["_time_numeric_"].mean()
            return f"🍔 Order type durations:\n{avg.to_string()}"

    if "festival" in q:
        if festival_col:
            avg = df.groupby(festival_col)["_time_numeric_"].mean()
            return f"🎉 Festival vs Non-Festival:\n{avg.to_string()}"

    if "traffic" in q:
        if traffic_col:
            avg = df.groupby(traffic_col)["_time_numeric_"].mean()
            return f"🚦 Traffic impact:\n{avg.to_string()}"

    if "weather" in q:
        if weather_col:
            avg = df.groupby(weather_col)["_time_numeric_"].mean()
            return f"☁️ Weather impact:\n{avg.to_string()}"

    if "age" in q:
        if age_col:
            corr = pd.to_numeric(df[age_col], errors="coerce").corr(df["_time_numeric_"])
            return f"👤 Age vs Time correlation: {corr:.2f}"

    if "vehicle condition" in q:
        if vehicle_cond_col:
            avg = df.groupby(vehicle_cond_col)["_time_numeric_"].mean()
            return f"🛵 Vehicle condition impact:\n{avg.to_string()}"

    if "distance" in q:
        if not df["_distance_km_"].isna().all():
            corr = df["_distance_km_"].corr(df["_time_numeric_"])
            return f"📍 Distance vs Time correlation: {corr:.2f}"

    if "area" in q or ("city" in q and "delay" in q):
        if city_col:
            avg = df.groupby(city_col)["_time_numeric_"].mean().sort_values(ascending=False)
            return f"⏳ Areas with highest delays:\n{avg.to_string()}"

    # -------- Fallback --------
    return "❓ I couldn't find an exact answer. Try rephrasing (e.g. 'average delivery per city', 'impact of traffic', 'order type durations')."

# --------------------------------------
# App start
# --------------------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")
st.markdown("Ask questions about your delivery dataset and get clean, data-driven answers.")

# Load dataset directly
DATA_FILE = "Zomato Dataset.csv"
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found! Please make sure 'Zomato Dataset.csv' is in the repo.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"❌ Could not read dataset: {e}")
    st.stop()

# Sidebar
col_left, col_right = st.columns([1, 3])
with col_left:
    st.subheader("📝 Options")
    with st.expander("📊 Delivery Performance"):
        st.markdown("""
        - Which delivery person is the fastest on average?  
        - What is the average delivery time per city?  
        - How do multiple deliveries affect delivery time?  
        - Which vehicle type is most efficient for deliveries?  
        """)
    with st.expander("👥 Customer & Order Insights"):
        st.markdown("""
        - What types of orders take the longest to deliver?  
        - Does order time affect delivery speed?  
        - Are deliveries slower during festivals?  
        """)
    with st.expander("🌍 Environment & External Factors"):
        st.markdown("""
        - How does traffic density impact delivery time?  
        - Do weather conditions affect delivery speed?  
        - How do restaurant vs. delivery locations affect time?  
        """)
    with st.expander("🚴 Delivery Personnel Metrics"):
        st.markdown("""
        - Who has the highest ratings and fastest deliveries?  
        - Does the age of the delivery person affect speed?  
        - Does vehicle condition impact delivery time?  
        """)
    with st.expander("📍 Geospatial Insights"):
        st.markdown("""
        - Which areas have the highest delays?  
        - What is the correlation between distance and time?  
        """)
    st.markdown("---\n💡 Tip: Try typing your own custom questions!")

# Right panel
with col_right:
    st.subheader("🤖 AI Response")
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = ""
    answer_box = st.text_area(" ", value=st.session_state["last_answer"], height=160)

    st.subheader("🔍 Enter your question")
    q_val = st.text_area("", value=st.session_state.get("question", ""), key="question_input", height=80)

    if st.button("🚀 Ask Question"):
        if not q_val.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            answer = chat_with_delivery_data(q_val, df)
            st.session_state["last_answer"] = answer
            st.session_state["question"] = q_val
