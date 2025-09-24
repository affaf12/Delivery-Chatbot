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
    # try case-insensitive match
    lowcols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowcols:
            return lowcols[c.lower()]
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in kilometers between two lat/lon points
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
    rating_col = find_col(df, ["Delivery_person_Ratings", "Delivery_person_Rating", "Delivery_person_Ratings"])
    time_col = find_col(df, ["Time_taken (min)", "Time_taken_min", "Time_taken", "Time_taken_minutes"])
    city_col = find_col(df, ["City", "city", "Area"])
    multi_col = find_col(df, ["multiple_deliveries", "Multiple_deliveries", "multiple_deliveries_flag"])
    vehicle_col = find_col(df, ["Type_of_vehicle", "vehicle_type", "Type_of_vehicle"])
    order_type_col = find_col(df, ["Type_of_order", "type_of_order", "Order_Type"])
    time_order_col = find_col(df, ["Time_Orderd", "Time_Orderd", "Time_Order_picked", "Order_Time"])
    weather_col = find_col(df, ["Weather_conditions", "Weather", "weather_conditions"])
    traffic_col = find_col(df, ["Road_traffic_density", "Traffic", "road_traffic_density"])
    festival_col = find_col(df, ["Festival", "festival"])
    rest_lat = find_col(df, ["Restaurant_latitude", "restaurant_latitude", "Restaurant_Latitude"])
    rest_lon = find_col(df, ["Restaurant_longitude", "restaurant_longitude", "Restaurant_Longitude"])
    del_lat = find_col(df, ["Delivery_location_latitude", "delivery_location_latitude", "Delivery_Location_Latitude"])
    del_lon = find_col(df, ["Delivery_location_longitude", "delivery_location_longitude", "Delivery_Location_Longitude"])
    age_col = find_col(df, ["Delivery_person_Age", "Delivery_person_age", "Age"])
    vehicle_cond_col = find_col(df, ["Vehicle_condition", "Vehicle_Condition", "vehicle_condition"])

    # ensure numeric time column exists for many queries
    if time_col:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = pd.NA

    # create distance column if lat/lon available
    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # ----- Queries -----
    # 1) Highest rating
    if "highest rating" in q or "highest rated" in q or ("highest" in q and "rating" in q):
        if rating_col and id_col:
            best = df.loc[pd.to_numeric(df[rating_col], errors="coerce").idxmax()]
            return f"✅ Highest rating: {best.get(id_col,'N/A')} — rating {best.get(rating_col)}"
        return "I can't find rating or ID columns. Available columns: " + ", ".join(df.columns[:30])

    # 2) Fastest on average
    if "fastest on average" in q or ("fastest" in q and "average" in q):
        if id_col and not df["_time_numeric_"].isna().all():
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            if avg_times.empty:
                return "No valid numeric time data to compute averages."
            fastest = avg_times.idxmin()
            return f"⚡ Fastest on average: {fastest} with {avg_times.min():.2f} minutes (avg)"
        return "Missing delivery ID or numeric time column."

    # 3) Average delivery time per city
    if "average delivery time per city" in q or ("average" in q and "per city" in q) or ("per city" in q):
        if city_col and not df["_time_numeric_"].isna().all():
            city_avg = df.groupby(city_col)["_time_numeric_"].mean().sort_values()
            return "Average delivery time by city (minutes):\n" + city_avg.round(2).to_string()
        return "City column or numeric time missing."

    # 4) Multiple deliveries effect
    if "multiple deliveries" in q or ("multiple" in q and "deliver" in q):
        if multi_col and not df["_time_numeric_"].isna().all():
            res = df.groupby(multi_col)["_time_numeric_"].mean().round(2)
            return "Average time by multiple_deliveries:\n" + res.to_string()
        return "Column for multiple deliveries or numeric time is missing."

    # 5) Vehicle efficiency
    if "vehicle type" in q and ("efficient" in q or "efficient" in q or "most efficient" in q):
        if vehicle_col and not df["_time_numeric_"].isna().all():
            veh = df.groupby(vehicle_col)["_time_numeric_"].mean().round(2)
            best = veh.idxmin()
            return f"🚗 Most efficient vehicle: {best} — {veh.min():.2f} min avg\n\nAll vehicle averages:\n{veh.to_string()}"
        return "Vehicle type or numeric time column missing."

    # 6) Which order types take longest
    if ("types of orders" in q or "order type" in q or "which order" in q) and "longest" in q:
        if order_type_col and not df["_time_numeric_"].isna().all():
            order_avg = df.groupby(order_type_col)["_time_numeric_"].mean().round(2)
            slowest = order_avg.idxmax()
            return f"🍔 Slowest order type: {slowest} — {order_avg.max():.2f} min avg\n\nDetails:\n{order_avg.to_string()}"
        return "Order type or numeric time column missing."

    # 7) Does order time affect speed (morning/evening)
    if ("order time" in q or "time of order" in q or "morning" in q or "evening" in q) and time_order_col:
        try:
            t = pd.to_datetime(df[time_order_col], errors="coerce")
            hours = t.dt.hour
            bins = {
                "night": hours[(hours >= 0) & (hours <= 5)],
                "morning": hours[(hours >= 6) & (hours <= 11)],
                "afternoon": hours[(hours >= 12) & (hours <= 17)],
                "evening": hours[(hours >= 18) & (hours <= 23)]
            }
            # compute mean time per bin
            res = {}
            for name, idxs in bins.items():
                if len(idxs) == 0:
                    res[name] = np.nan
                else:
                    res[name] = df.loc[idxs.index, "_time_numeric_"].mean()
            res_ser = pd.Series(res).round(2).sort_values()
            return "Average delivery time by period (minutes):\n" + res_ser.to_string()
        except Exception:
            return "Could not parse order times. Column exists but format is not recognized."

    # 8) Festival effect
    if "festival" in q:
        if festival_col and not df["_time_numeric_"].isna().all():
            res = df.groupby(festival_col)["_time_numeric_"].mean().round(2)
            return "Avg delivery time by festival flag:\n" + res.to_string()
        return "No Festival column or numeric time column."

    # 9) Traffic density impact
    if "traffic" in q or "road_traffic_density" in q:
        if traffic_col and not df["_time_numeric_"].isna().all():
            res = df.groupby(traffic_col)["_time_numeric_"].mean().round(2).sort_values()
            return "Avg delivery time by traffic density:\n" + res.to_string()
        return "Traffic column or numeric time missing."

    # 10) Weather conditions impact
    if "weather" in q:
        if weather_col and not df["_time_numeric_"].isna().all():
            res = df.groupby(weather_col)["_time_numeric_"].mean().round(2).sort_values()
            return "Avg delivery time by weather condition:\n" + res.to_string()
        return "Weather column or numeric time missing."

    # 11) Restaurant vs delivery location distance effect
    if ("distance" in q and "time" in q) or ("restaurant" in q and "delivery" in q and "location" in q):
        if not df["_distance_km_"].isna().all() and not df["_time_numeric_"].isna().all():
            corr = df[["_distance_km_", "_time_numeric_"]].corr().iloc[0,1]
            corr_val = np.nan if pd.isna(corr) else round(float(corr), 2)
            return f"📏 Distance vs time correlation (Pearson): {corr_val}\n(positive means longer distance ~ longer time)"
        return "Latitude/longitude columns or numeric time missing to compute distance/time correlation."

    # 12) Highest ratings + fastest deliveries (composite)
    if "highest ratings and fastest" in q or ("highest ratings" in q and "fastest" in q) or ("best performer" in q):
        if id_col and rating_col and not df["_time_numeric_"].isna().all():
            avg_times = df.groupby(id_col)["_time_numeric_"].mean()
            avg_ratings = pd.to_numeric(df.groupby(id_col)[rating_col].mean(), errors='coerce')
            score = (avg_ratings.fillna(0) / (avg_times.replace(0, np.nan))).dropna()
            if score.empty:
                return "Not enough numeric rating/time data to compute performance."
            best = score.idxmax()
            return f"🏆 Best performer (rating/time): {best}\nTop 3 performers:\n{score.sort_values(ascending=False).head(3).round(3).to_string()}"
        return "Missing ID, rating, or time numeric columns."

    # 13) Correlation age vs speed
    if "age" in q and age_col:
        if not df["_time_numeric_"].isna().all():
            age_numeric = pd.to_numeric(df[age_col], errors="coerce")
            corr = age_numeric.corr(df["_time_numeric_"])
            return f"📊 Correlation between delivery person age and delivery time: {None if pd.isna(corr) else round(float(corr),2)}"
        return "Time column missing for correlation."

    # 14) Vehicle condition impact
    if "vehicle condition" in q or "vehicle_condition" in q:
        if vehicle_cond_col and not df["_time_numeric_"].isna().all():
            res = df.groupby(vehicle_cond_col)["_time_numeric_"].mean().round(2)
            return "Avg delivery time by vehicle condition:\n" + res.to_string()
        return "Vehicle condition or time column missing."

    # 15) Areas with highest delays
    if "areas" in q and ("delay" in q or "delays" in q or "highest delays" in q):
        if city_col and not df["_time_numeric_"].isna().all():
            city_avg = df.groupby(city_col)["_time_numeric_"].mean().sort_values(ascending=False).head(10).round(2)
            return "Top areas by avg delivery time (minutes):\n" + city_avg.to_string()
        return "City or time column missing."

    # 16) distance vs time correlation (explicit)
    if "distance vs time" in q or "distance correlation" in q:
        if not df["_distance_km_"].isna().all() and not df["_time_numeric_"].isna().all():
            corr = df["_distance_km_"].corr(df["_time_numeric_"])
            return f"📏 Distance vs time correlation: {None if pd.isna(corr) else round(float(corr),2)}"
        return "Missing distance (lat/lon) or numeric time column."

    # --- fallback: short meta answers or summary statistics ---
    if "average rating" in q:
        if rating_col:
            return f"⭐ Average delivery person rating: {pd.to_numeric(df[rating_col], errors='coerce').mean():.2f}"
        return "Rating column not found."

    if "youngest" in q or "oldest" in q:
        if age_col and id_col:
            if "youngest" in q:
                idx = pd.to_numeric(df[age_col], errors='coerce').idxmin()
            else:
                idx = pd.to_numeric(df[age_col], errors='coerce').idxmax()
            row = df.loc[idx]
            return f"{'Youngest' if 'youngest' in q else 'Oldest'}: {row.get(id_col)} — age {row.get(age_col)}"
        return "Age or ID column missing."

    # final fallback: ask to use one of the sample questions
    sample_qs = [
        "Which delivery person has the highest rating?",
        "Which delivery person is the fastest on average?",
        "What is the average delivery time per city?",
        "How does traffic density impact delivery time?",
        "Distance vs. time correlation for deliveries"
    ]
    return ("❓ I couldn't identify a direct match for that question.\n"
            "Try one of these example questions:\n- " + "\n- ".join(sample_qs))

# --------------------------------------
# App start: load dataset (safe)
# --------------------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")
st.markdown("Ask questions about your delivery dataset and get clean, data-driven answers.")

# Sidebar uploader / URL / preview
col_left, col_right = st.columns([1, 3])

with col_left:
    st.subheader("📝 Options")
    st.markdown("Upload CSV (optional) or paste raw GitHub CSV URL. If your CSV is present in the repo, the app will use it.")
    uploaded = st.file_uploader("Upload dataset CSV", type=["csv"])
    raw_url = st.text_input("Or paste raw GitHub CSV URL (optional)", "")

    # Expandable categories (same content as requested)
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

# load dataset from repo file if exists, else uploaded file, else raw_url
DATA_FILE = "Zomato Dataset.csv"
df = None
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        st.warning("Could not read local file 'Zomato Dataset.csv'. Try uploading or pasting the raw URL.")
if df is None and uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Uploaded file could not be read as CSV.")
if df is None and raw_url.strip():
    try:
        df = pd.read_csv(raw_url.strip())
    except Exception:
        st.error("Could not load CSV from provided URL. Make sure it's a raw GitHub link or direct CSV URL.")

if df is None:
    st.warning("No dataset loaded yet. Upload a CSV in the left panel or add a raw GitHub CSV URL.")
    st.stop()

# show a compact preview and columns
with col_right:
    st.subheader("Dataset preview")
    st.dataframe(df.head(5))
    st.markdown("**Columns:** " + ", ".join(df.columns.astype(str)))

# --------------------------------------
# Right panel: Answer at top -> question -> button
# --------------------------------------
with col_right:
    st.subheader("🤖 AI Response")
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = ""

    answer_box = st.text_area(" ", value=st.session_state["last_answer"], height=160)

    st.subheader("🔍 Enter your question")
    examples = [
        "",
        "Which delivery person has the highest rating?",
        "Which delivery person is the fastest on average?",
        "What is the average delivery time per city?",
        "How does traffic density impact delivery time?",
        "Distance vs. time correlation for deliveries",
        "Which areas have the highest delays?"
    ]
    sel = st.selectbox("Quick examples (select to load into question box)", examples, index=0)
    if sel:
        st.session_state["question"] = sel

    q_val = st.text_area("", value=st.session_state.get("question", ""), key="question_input", height=80)

    if st.button("🚀 Ask Question"):
        if not q_val.strip():
            st.warning("⚠️ Please enter a question (or choose an example).")
        else:
            # run Q/A function
            answer = chat_with_delivery_data(q_val, df)
            st.session_state["last_answer"] = answer
            st.session_state["question"] = q_val
            # update visible answer box
            st.experimental_rerun()
