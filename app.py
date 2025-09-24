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
# Core Q/A
# -------------------------
def chat_with_delivery_data(question: str, df: pd.DataFrame):
    q = (question or "").lower().strip()

    # map columns flexibly
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
    age_col = find_col(df, ["Delivery_person_Age", "Delivery_person_age", "Age"])
    vehicle_cond_col = find_col(df, ["Vehicle_condition", "vehicle_condition"])

    # numeric delivery time
    if time_col in df.columns:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = np.nan

    # distance
    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # -------------------------
    # Answering patterns (simplified)
    # -------------------------
    if "fastest" in q and id_col:
        avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
        if avg_times.empty:
            return "No valid numeric delivery-time data."
        fastest_id = avg_times.idxmin()
        fastest_val = avg_times.min()
        return f"⚡ Fastest: **{fastest_id}** — {fastest_val:.2f} min"

    if "highest rating" in q and rating_col and id_col:
        nums = pd.to_numeric(df[rating_col], errors="coerce")
        if nums.dropna().empty:
            return "No numeric rating data available."
        idx_max = nums.idxmax()
        best_row = df.loc[idx_max]
        return f"✅ Highest rating: **{best_row[id_col]}** — {best_row[rating_col]} ⭐"

    if "average" in q and "city" in q and city_col:
        return df.groupby(city_col)["_time_numeric_"].mean().reset_index()

    if "multiple deliveries" in q and multi_col:
        return df.groupby(multi_col)["_time_numeric_"].mean().reset_index()

    if "vehicle type" in q and vehicle_col:
        return df.groupby(vehicle_col)["_time_numeric_"].mean().reset_index()

    if "order" in q and order_type_col:
        return df.groupby(order_type_col)["_time_numeric_"].mean().reset_index()

    if "festival" in q and festival_col:
        return df.groupby(festival_col)["_time_numeric_"].mean().reset_index()

    if "traffic" in q and traffic_col:
        return df.groupby(traffic_col)["_time_numeric_"].mean().reset_index()

    if "weather" in q and weather_col:
        return df.groupby(weather_col)["_time_numeric_"].mean().reset_index()

    if "age" in q and age_col:
        corr = pd.to_numeric(df[age_col], errors="coerce").corr(df["_time_numeric_"])
        return f"👤 Correlation between age and time: **{corr:.2f}**"

    if "vehicle condition" in q and vehicle_cond_col:
        return df.groupby(vehicle_cond_col)["_time_numeric_"].mean().reset_index()

    if "distance" in q:
        if not df["_distance_km_"].isna().all():
            corr = df["_distance_km_"].corr(df["_time_numeric_"])
            return f"📍 Correlation between distance and time: **{corr:.2f}**"

    return "❓ I couldn't answer. Try one of the example questions."

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")

DATA_FILE = "Zomato Dataset.csv"
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found. Put 'Zomato Dataset.csv' in this folder.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"❌ Could not read dataset: {e}")
    st.stop()

col_left, col_right = st.columns([1, 3])

# -------------------------
# LEFT PANEL: Stakeholder Questions
# -------------------------
with col_left:
    st.subheader("📝 Types of Questions Stakeholders Might Ask")

    # Delivery Performance
    with st.expander("📊 Delivery Performance", expanded=True):
        for q in [
            "Which delivery person is the fastest on average?",
            "What is the average delivery time per city?",
            "How do multiple deliveries affect delivery time?",
            "Which vehicle type is most efficient for deliveries?",
        ]:
            if st.button(q, key=f"btn_perf_{q}"):
                st.session_state["question"] = q
                st.session_state["last_answer"] = chat_with_delivery_data(q, df)
                st.rerun()

    # Customer & Order Insights
    with st.expander("👥 Customer & Order Insights"):
        for q in [
            "What types of orders take the longest to deliver?",
            "Does order time affect delivery speed?",
            "Are deliveries slower during festivals?",
        ]:
            if st.button(q, key=f"btn_cust_{q}"):
                st.session_state["question"] = q
                st.session_state["last_answer"] = chat_with_delivery_data(q, df)
                st.rerun()

    # Environmental & External Factors
    with st.expander("🌍 Environmental & External Factors"):
        for q in [
            "How does traffic density impact delivery time?",
            "Do weather conditions affect delivery speed?",
            "How do restaurant locations vs. delivery locations affect delivery time?",
        ]:
            if st.button(q, key=f"btn_env_{q}"):
                st.session_state["question"] = q
                st.session_state["last_answer"] = chat_with_delivery_data(q, df)
                st.rerun()

    # Delivery Personnel Metrics
    with st.expander("🚴 Delivery Personnel Metrics"):
        for q in [
            "Who has the highest ratings and fastest deliveries?",
            "Does the age of the delivery person correlate with speed or ratings?",
            "Does vehicle condition impact delivery time?",
        ]:
            if st.button(q, key=f"btn_person_{q}"):
                st.session_state["question"] = q
                st.session_state["last_answer"] = chat_with_delivery_data(q, df)
                st.rerun()

    # Geospatial Insights
    with st.expander("📍 Geospatial Insights"):
        for q in [
            "Which areas have the highest delays?",
            "Distance vs. time correlation for deliveries",
        ]:
            if st.button(q, key=f"btn_geo_{q}"):
                st.session_state["question"] = q
                st.session_state["last_answer"] = chat_with_delivery_data(q, df)
                st.rerun()

# -------------------------
# RIGHT PANEL: Responses
# -------------------------
with col_right:
    response_container = st.container()
    last_answer = st.session_state.get("last_answer", None)

    if has_answer(last_answer):
        response_container.markdown("### 🤖 AI Response")
        if isinstance(last_answer, pd.DataFrame):
            response_container.dataframe(last_answer, use_container_width=True)
        else:
            response_container.success(last_answer)

    with st.form(key="qa_form"):
        st.markdown("### 🔍 Enter your question")
        q_val = st.text_area(
            "Type here...",
            value=st.session_state.get("question", ""),
            key="question_input",
            height=80,
        )
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submitted = st.form_submit_button("🚀 Ask Question")
        with col_clear:
            cleared = st.form_submit_button("✖ Clear")

        if cleared:
            st.session_state["question"] = ""
            st.session_state["last_answer"] = None
            st.rerun()

        if submitted:
            if not q_val.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("Analyzing..."):
                    answer = chat_with_delivery_data(q_val, df)
                st.session_state["question"] = q_val
                st.session_state["last_answer"] = answer
                st.rerun()
