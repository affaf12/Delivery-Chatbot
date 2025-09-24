# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

# -------------------------
# Utility helpers
# -------------------------
def find_col(df, candidates):
    """Return first matching column name from candidates (case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lowcols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowcols:
            return lowcols[c.lower()]
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    """Return haversine distance in km or np.nan for bad values."""
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

def has_answer(val):
    """Return True if val is a non-empty answer (handles str/DataFrame/None)."""
    if val is None:
        return False
    if isinstance(val, pd.DataFrame):
        return not val.empty
    if isinstance(val, (list, tuple, set)):
        return len(val) > 0
    try:
        return str(val).strip() != ""
    except Exception:
        return True

# -------------------------
# Core Q/A (returns str or DataFrame)
# -------------------------
def chat_with_delivery_data(question: str, df: pd.DataFrame):
    q = (question or "").lower().strip()

    # column mapping
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
    rest_lat = find_col(df, ["Restaurant_latitude", "restaurant_latitude"])
    rest_lon = find_col(df, ["Restaurant_longitude", "restaurant_longitude"])
    del_lat = find_col(df, ["Delivery_location_latitude", "delivery_location_latitude"])
    del_lon = find_col(df, ["Delivery_location_longitude", "delivery_location_longitude"])
    age_col = find_col(df, ["Delivery_person_Age", "Age"])
    vehicle_cond_col = find_col(df, ["Vehicle_condition", "vehicle_condition"])

    # numeric time column
    if time_col:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = np.nan

    # distance column
    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # ---------- Rules ----------
    if "highest rating" in q:
        if rating_col and id_col:
            nums = pd.to_numeric(df[rating_col], errors="coerce")
            if nums.dropna().empty:
                return "No numeric rating data."
            idx_max = nums.idxmax()
            best_row = df.loc[idx_max]
            return f"✅ Highest rating: **{best_row.get(id_col)}** — {best_row.get(rating_col)} ⭐"

    if "fastest" in q:
        if id_col:
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            if avg_times.empty:
                return "No valid numeric delivery-time data."
            fastest_id = avg_times.idxmin()
            fastest_val = avg_times.min()
            return f"⚡ Fastest: **{fastest_id}** — {fastest_val:.2f} min"

    if "average" in q and "city" in q:
        if city_col:
            out = df.groupby(city_col)["_time_numeric_"].mean().reset_index()
            out["_time_numeric_"] = out["_time_numeric_"].round(2)
            return out

    if "traffic" in q:
        if traffic_col:
            out = df.groupby(traffic_col)["_time_numeric_"].mean().reset_index()
            out["_time_numeric_"] = out["_time_numeric_"].round(2)
            return out

    if "weather" in q:
        if weather_col:
            out = df.groupby(weather_col)["_time_numeric_"].mean().reset_index()
            out["_time_numeric_"] = out["_time_numeric_"].round(2)
            return out

    if "age" in q:
        if age_col:
            age_num = pd.to_numeric(df[age_col], errors="coerce")
            corr = age_num.corr(df["_time_numeric_"])
            return f"👤 Correlation (age vs delivery time): **{(round(corr,2) if not np.isnan(corr) else 'NaN')}**"

    if "vehicle condition" in q:
        if vehicle_cond_col:
            out = df.groupby(vehicle_cond_col)["_time_numeric_"].mean().reset_index()
            out["_time_numeric_"] = out["_time_numeric_"].round(2)
            return out

    if "distance" in q:
        if not df["_distance_km_"].isna().all():
            corr = df["_distance_km_"].corr(df["_time_numeric_"])
            return f"📍 Correlation (distance vs time): **{(round(corr,2) if not np.isnan(corr) else 'NaN')}**"

    return "❓ I couldn't find an exact answer. Try: *'fastest delivery person'*, *'average per city'*, *'impact of traffic'*."

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")

DATA_FILE = "Zomato Dataset.csv"
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found! Please put 'Zomato Dataset.csv' in the repo.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"❌ Could not read dataset: {e}")
    st.stop()

col_left, col_right = st.columns([1, 3])

with col_left:
    st.subheader("📝 Options")
    with st.expander("📊 Delivery Performance"):
        st.markdown("- Which delivery person is the fastest on average?\n- What is the average delivery time per city?")
    with st.expander("🌍 Environment & External Factors"):
        st.markdown("- How does traffic density impact delivery time?\n- Do weather conditions affect delivery speed?")
    with st.expander("🚴 Delivery Personnel Metrics"):
        st.markdown("- Who has the highest ratings and fastest deliveries?\n- Does the age of the delivery person affect speed?\n- Does vehicle condition impact delivery time?")
    with st.expander("📍 Geospatial Insights"):
        st.markdown("- Which areas have the highest delays?\n- What is the correlation between distance and time?")

with col_right:
    # ✅ Show AI response FIRST
    last_answer = st.session_state.get("last_answer", None)
    if has_answer(last_answer):
        st.subheader("🤖 AI Response")
        if isinstance(last_answer, pd.DataFrame):
            st.dataframe(last_answer, use_container_width=True)
        else:
            st.success(last_answer)

    # Input comes AFTER response
    st.subheader("🔍 Enter your question")
    q_val = st.text_area("", value=st.session_state.get("question", ""), key="question_input", height=80)

    if st.button("🚀 Ask Question"):
        if not q_val.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            answer = chat_with_delivery_data(q_val, df)
            st.session_state["question"] = q_val
            st.session_state["last_answer"] = answer
            st.rerun()   # ✅ fixed for latest Streamlit
