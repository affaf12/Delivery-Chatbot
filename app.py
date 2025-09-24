# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

# -------------------------
# Helpers
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
    """Haversine distance in kilometers, returns np.nan on bad input."""
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
# Core Q/A function (returns either str or pandas.DataFrame)
# -------------------------
def chat_with_delivery_data(question, df):
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
    age_col = find_col(df, ["Delivery_person_Age", "Delivery_person_age", "Age"])
    vehicle_cond_col = find_col(df, ["Vehicle_condition", "vehicle_condition"])

    # prepare numeric time column
    if time_col in df.columns:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = np.nan

    # compute distance if lat/lon available
    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # --- Queries (return DataFrame or string) ---

    # Highest rating
    if "highest rating" in q or ("highest" in q and "rating" in q):
        if rating_col and id_col:
            nums = pd.to_numeric(df[rating_col], errors="coerce")
            if nums.dropna().empty:
                return "No numeric rating data available."
            idx = nums.idxmax()
            row = df.loc[idx]
            return f"✅ Highest rating: **{row.get(id_col,'N/A')}** — {row.get(rating_col)} ⭐"
        return "Rating or ID column not found in dataset."

    # Fastest on average (per delivery person)
    if "fastest" in q and ("average" in q or "on average" in q or "on avg" in q):
        if id_col:
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            if avg_times.empty:
                return "No valid numeric time data to compute averages."
            fastest_id = avg_times.idxmin()
            fastest_val = avg_times.min()
            return f"⚡ Fastest: **{fastest_id}** — {fastest_val:.2f} minutes (avg)"
        return "Delivery person ID column not found."

    # Average delivery time per city
    if "average" in q and "city" in q:
        if city_col:
            out = df.groupby(city_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "City column or time column missing."

    # Multiple deliveries effect
    if "multiple deliveries" in q or ("multiple" in q and "deliver" in q):
        if multi_col:
            out = df.groupby(multi_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out
        return "Multiple deliveries column not found."

    # Vehicle type efficiency
    if "vehicle" in q and ("efficient" in q or "efficiency" in q or "most efficient" in q):
        if vehicle_col:
            out = df.groupby(vehicle_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "Vehicle type column not found."

    # Order type durations
    if ("order type" in q or "types of orders" in q or "order took the longest" in q or ("longest" in q and "order" in q)):
        if order_type_col:
            out = df.groupby(order_type_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min", ascending=False)
        return "Order type column not found."

    # Festival effect
    if "festival" in q:
        if festival_col:
            out = df.groupby(festival_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out
        return "Festival column not found."

    # Traffic impact
    if "traffic" in q:
        if traffic_col:
            out = df.groupby(traffic_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "Traffic column not found."

    # Weather impact
    if "weather" in q:
        if weather_col:
            out = df.groupby(weather_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "Weather column not found."

    # Age correlation
    if "age" in q:
        if age_col and not df["_time_numeric_"].isna().all():
            age_num = pd.to_numeric(df[age_col], errors="coerce")
            if age_num.dropna().empty:
                return "No numeric age data available."
            corr = age_num.corr(df["_time_numeric_"])
            return f"👤 Correlation (age vs time): **{(round(corr,2) if not np.isnan(corr) else 'NaN')}**"
        return "Age or time column not found."

    # Vehicle condition
    if "vehicle condition" in q or "vehicle_condition" in q:
        if vehicle_cond_col:
            out = df.groupby(vehicle_cond_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "Vehicle condition column not found."

    # Distance vs time correlation
    if ("distance" in q and "time" in q) or ("distance" in q and "correlation" in q):
        if not df["_distance_km_"].isna().all() and not df["_time_numeric_"].isna().all():
            corr = df["_distance_km_"].corr(df["_time_numeric_"])
            return f"📍 Correlation (distance km vs time min): **{(round(corr,2) if not np.isnan(corr) else 'NaN')}**"
        return "Latitude/longitude or time data missing to compute distance/time correlation."

    # Areas with highest delays
    if ("areas" in q and ("delay" in q or "delays" in q)) or ("highest delays" in q):
        if city_col:
            out = df.groupby(city_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min", ascending=False).head(10)
        return "City or time column not found."

    # fallback
    return ("❓ I couldn't identify a direct match for that question.\n"
            "Try: 'Which delivery person has the highest rating?',\n"
            "'Which delivery person is the fastest on average?',\n"
            "'What is the average delivery time per city?', or\n"
            "'How does traffic density impact delivery time?'")

# -------------------------
# App start / UI
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")
st.markdown("Ask questions about your delivery dataset and get clean, data-driven answers.")

# load dataset from repo
DATA_FILE = "Zomato Dataset.csv"
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found! Please make sure 'Zomato Dataset.csv' is in the repo root.")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"❌ Could not read dataset: {e}")
    st.stop()

# layout: sidebar (options) + main panel
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
    st.markdown("---\n💡 Tip: Ask plain English questions like the bullets above.")

# RIGHT: response placeholder (top) + question form (below)
with col_right:
    response_placeholder = st.empty()  # will render the answer above the form

    # show last answer (if any) safely
    last = st.session_state.get("last_answer", None)
    if has_answer(last):
        response_placeholder.subheader("🤖 AI Response (latest)")
        if isinstance(last, pd.DataFrame):
            response_placeholder.dataframe(last, use_container_width=True)
        else:
            response_placeholder.markdown(last)

    # form for question input
    with st.form(key="qa_form", clear_on_submit=False):
        st.subheader("🔍 Enter your question")
        q_val = st.text_area("", value=st.session_state.get("question", ""), key="question_input", height=80)
        submitted = st.form_submit_button("🚀 Ask Question")

        if submitted:
            if not q_val.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                answer = chat_with_delivery_data(q_val, df)
                st.session_state["question"] = q_val
                st.session_state["last_answer"] = answer

                # update immediately in the placeholder (same run)
                response_placeholder.subheader("🤖 AI Response")
                if isinstance(answer, pd.DataFrame):
                    response_placeholder.dataframe(answer, use_container_width=True)
                else:
                    response_placeholder.markdown(answer)
