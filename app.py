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

    # column mapping (robust to capitalization)
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

    # make a numeric time column for computations (safe)
    if time_col in df.columns:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = np.nan

    # distance column if lat/lon present
    if rest_lat and rest_lon and del_lat and del_lon:
        # only compute once per run (safe)
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # ---------- Rules / Intent handlers ----------
    # Highest rating (single best + top-3 table)
    if "highest rating" in q or ("highest" in q and "rating" in q):
        if rating_col and id_col:
            nums = pd.to_numeric(df[rating_col], errors="coerce")
            if nums.dropna().empty:
                return "No numeric rating data available."
            idx_max = nums.idxmax()
            best_row = df.loc[idx_max]
            # also prepare top 3
            top3 = df.assign(_r = pd.to_numeric(df[rating_col], errors="coerce")) \
                     .sort_values("_r", ascending=False).head(5)[[id_col, rating_col]]
            top3 = top3.dropna()
            return f"✅ Highest rating: **{best_row.get(id_col, 'N/A')}** — {best_row.get(rating_col)} ⭐\n\nTop performers (by rating):", top3

    # Fastest on average (per delivery person) — returns string + small DF
    if "fastest" in q and ("average" in q or "on average" in q or "on avg" in q):
        if id_col:
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            if avg_times.empty:
                return "No valid numeric delivery-time data."
            fastest_id = avg_times.idxmin()
            fastest_val = avg_times.min()
            top3 = avg_times.sort_values().head(5).reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            top3["avg_time_min"] = top3["avg_time_min"].round(2)
            return f"⚡ Fastest delivery person (avg): **{fastest_id}** — {fastest_val:.2f} min (avg)\n\nTop (by avg time):", top3

    # Average delivery time per city (returns DataFrame sorted ascending)
    if "average" in q and "city" in q:
        if city_col:
            out = df.groupby(city_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "City or time column not found."

    # Multiple deliveries effect
    if "multiple deliveries" in q or ("multiple" in q and "deliver" in q):
        if multi_col:
            out = df.groupby(multi_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out
        return "multiple_deliveries column not found."

    # Vehicle efficiency (returns DataFrame sorted ascending)
    if "vehicle" in q and ("efficient" in q or "efficiency" in q or "most efficient" in q):
        if vehicle_col:
            out = df.groupby(vehicle_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "vehicle type column not found."

    # Order type durations (which type takes longest)
    if ("order type" in q or "types of orders" in q or ("longest" in q and "order" in q) or "which order" in q):
        if order_type_col:
            out = df.groupby(order_type_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min", ascending=False)
        return "Type_of_order column not found."

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
        return "Road_traffic_density column not found."

    # Weather impact
    if "weather" in q:
        if weather_col:
            out = df.groupby(weather_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "Weather_conditions column not found."

    # Age correlation
    if "age" in q:
        if age_col and not df["_time_numeric_"].isna().all():
            age_num = pd.to_numeric(df[age_col], errors="coerce")
            if age_num.dropna().empty:
                return "No numeric age data."
            corr = age_num.corr(df["_time_numeric_"])
            return f"👤 Correlation (age vs delivery time): **{(round(corr,2) if not np.isnan(corr) else 'NaN')}**"
        return "Age or time column not found."

    # Vehicle condition impact
    if "vehicle condition" in q or "vehicle_condition" in q:
        if vehicle_cond_col:
            out = df.groupby(vehicle_cond_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min")
        return "Vehicle_condition column not found."

    # Distance vs time correlation
    if ("distance" in q and "time" in q) or ("distance" in q and "correlation" in q):
        if not df["_distance_km_"].isna().all() and not df["_time_numeric_"].isna().all():
            corr = df["_distance_km_"].corr(df["_time_numeric_"])
            return f"📍 Correlation (distance km vs time min): **{(round(corr,2) if not np.isnan(corr) else 'NaN')}**"
        return "Latitude/longitude or time data missing to compute correlation."

    # Areas with highest delays (top 10)
    if ("areas" in q and ("delay" in q or "delays" in q)) or ("highest delays" in q):
        if city_col:
            out = df.groupby(city_col)["_time_numeric_"].mean().reset_index().rename(columns={"_time_numeric_":"avg_time_min"})
            out["avg_time_min"] = out["avg_time_min"].round(2)
            return out.sort_values("avg_time_min", ascending=False).head(10)
        return "City or time column not found."

    # Final fallback (helpful examples)
    return ("❓ I couldn't identify a direct match for that question.\n\n"
            "Try one of these example queries:\n"
            "- Which delivery person has the highest rating?\n"
            "- Which delivery person is the fastest on average?\n"
            "- What is the average delivery time per city?\n"
            "- How does traffic density impact delivery time?\n"
            )

# -------------------------
# UI: Streamlit app
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")
st.write("Ask plain-English questions about your delivery dataset. Results appear above the question box.")

# load dataset (must be present in repo)
DATA_FILE = "Zomato Dataset.csv"
if not os.path.exists(DATA_FILE):
    st.error("❌ Dataset not found. Put 'Zomato Dataset.csv' in the app folder (repo root).")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"❌ Failed to read dataset: {e}")
    st.stop()

# build layout: sidebar + main
col_left, col_right = st.columns([1, 3])

# ------- Sidebar: categories + clickable example questions -------
with col_left:
    st.subheader("📝 Options")
    st.markdown("Click any example to auto-fill and run it.")

    # grouped examples per category (clickable)
    examples = {
        "Delivery Performance": [
            "Which delivery person is the fastest on average?",
            "Which delivery person has the highest rating?"
        ],
        "Customer & Order Insights": [
            "What types of orders take the longest to deliver?",
            "Does order time (morning/evening) affect delivery speed?"
        ],
        "Environment & External": [
            "How does traffic density impact delivery time?",
            "Do weather conditions affect delivery speed?"
        ],
        "Personnel Metrics": [
            "Does the age of the delivery person affect delivery time?",
            "Does vehicle condition impact delivery time?"
        ],
        "Geospatial Insights": [
            "Which areas have the highest delays?",
            "What is the correlation between distance and time?"
        ]
    }

    # show expanders and buttons
    for heading, qs in examples.items():
        with st.expander(heading):
            for q in qs:
                # each button sets question and computes answer immediately via callback
                if st.button(q, key=f"btn_{q}"):
                    st.session_state["question"] = q
                    # compute and store answer
                    st.session_state["last_answer"] = chat_with_delivery_data(q, df)
                    # prevent immediate rerun loop — use a flag to display (it will show due to same-run)
                    st.experimental_rerun()

    st.markdown("---")
    st.caption("Tip: you can also type your own question in the main panel below.")

# ------- Main panel: Response on top, question form below -------
with col_right:
    # placeholder area for answer
    response_container = st.container()

    # display last answer (if any) safely
    last_answer = st.session_state.get("last_answer", None)
    if has_answer(last_answer):
        response_container.markdown("### 🤖 AI Response (latest)")
        # our chat function sometimes returns a tuple (text, DataFrame) for certain queries
        if isinstance(last_answer, tuple) and len(last_answer) == 2 and isinstance(last_answer[1], pd.DataFrame):
            # tuple: (message, dataframe)
            response_container.markdown(last_answer[0])
            df_out = last_answer[1]
            response_container.dataframe(df_out, use_container_width=True)
            # chart if possible
            if df_out.shape[1] == 2 and pd.api.types.is_numeric_dtype(df_out.iloc[:,1]):
                try:
                    chart_data = df_out.set_index(df_out.columns[0]).iloc[:,0]
                    response_container.bar_chart(chart_data)
                except Exception:
                    pass
        elif isinstance(last_answer, pd.DataFrame):
            response_container.dataframe(last_answer, use_container_width=True)
            # auto-chart if the DF has one numeric column besides the index
            if last_answer.shape[1] == 2 and pd.api.types.is_numeric_dtype(last_answer.iloc[:,1]):
                try:
                    chart_data = last_answer.set_index(last_answer.columns[0]).iloc[:,0]
                    response_container.bar_chart(chart_data)
                except Exception:
                    pass
        else:
            # plain text answer
            response_container.markdown(last_answer)

    # input form below
    with st.form(key="qa_form", clear_on_submit=False):
        st.markdown("### 🔍 Enter your question")
        q_val = st.text_area("Type question here...", value=st.session_state.get("question", ""), key="question_input", height=80)
        col_submit, col_clear = st.columns([1,1])
        with col_submit:
            submitted = st.form_submit_button("🚀 Ask Question")
        with col_clear:
            cleared = st.form_submit_button("✖ Clear")

        if cleared:
            # clear only question & last_answer
            st.session_state["question"] = ""
            st.session_state["last_answer"] = None
            # refresh to update UI
            st.experimental_rerun()

        if submitted:
            if not q_val or not q_val.strip():
                st.warning("⚠️ Please enter a question (or click an example on the left).")
            else:
                with st.spinner("Thinking..."):
                    answer = chat_with_delivery_data(q_val, df)
                st.session_state["question"] = q_val
                st.session_state["last_answer"] = answer
                # update display immediately (same-run)
                st.experimental_rerun()
