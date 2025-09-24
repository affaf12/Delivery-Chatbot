# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
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
# Core Q/A (returns str or DataFrame or tuple)
# -------------------------
def chat_with_delivery_data(question: str, df: pd.DataFrame):
    q = (question or "").lower().strip()

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

    if time_col in df.columns:
        df["_time_numeric_"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["_time_numeric_"] = np.nan

    if rest_lat and rest_lon and del_lat and del_lon:
        df["_distance_km_"] = df.apply(
            lambda r: haversine_km(r[rest_lat], r[rest_lon], r[del_lat], r[del_lon]), axis=1
        )
    else:
        df["_distance_km_"] = np.nan

    # Highest rating
    if "highest rating" in q:
        if rating_col and id_col:
            nums = pd.to_numeric(df[rating_col], errors="coerce")
            if nums.dropna().empty:
                return "No numeric rating data available."
            idx_max = nums.idxmax()
            best_row = df.loc[idx_max]
            top3 = df.assign(_r=nums).sort_values("_r", ascending=False).head(5)[[id_col, rating_col]]
            return f"✅ Highest rating: **{best_row[id_col]}** — {best_row[rating_col]} ⭐", top3

    # Fastest on average
    if "fastest" in q:
        if id_col:
            avg_times = df.groupby(id_col)["_time_numeric_"].mean().dropna()
            if avg_times.empty:
                return "No valid numeric delivery-time data."
            fastest_id = avg_times.idxmin()
            fastest_val = avg_times.min()
            top3 = avg_times.sort_values().head(5).reset_index().rename(columns={"_time_numeric_": "avg_time_min"})
            top3["avg_time_min"] = top3["avg_time_min"].round(2)
            return f"⚡ Fastest on average: **{fastest_id}** — {fastest_val:.2f} min", top3

    # Average time per city
    if "average" in q and "city" in q:
        if city_col:
            out = df.groupby(city_col)["_time_numeric_"].mean().reset_index()
            out["avg_time_min"] = out["_time_numeric_"].round(2)
            return out[[city_col, "avg_time_min"]].sort_values("avg_time_min")

    return "❓ I couldn't find an exact answer. Try rephrasing or asking about ratings, fastest deliveries, city averages, etc."

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")
st.title("🚚 Delivery Data Chatbot")
st.caption("Ask questions about your delivery dataset. The AI Response will always appear above the question box.")

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

# Sidebar with examples
with col_left:
    st.subheader("📝 Example Questions")
    for q in [
        "Which delivery person has the highest rating?",
        "Which delivery person is the fastest on average?",
        "What is the average delivery time per city?"
    ]:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["question"] = q
            st.session_state["last_answer"] = chat_with_delivery_data(q, df)
            st.experimental_rerun()

# Main panel
with col_right:
    response_container = st.container()

    last_answer = st.session_state.get("last_answer", None)
    if has_answer(last_answer):
        response_container.markdown("### 🤖 AI Response")
        if isinstance(last_answer, tuple):
            msg, df_out = last_answer
            response_container.markdown(msg)
            response_container.dataframe(df_out, use_container_width=True)

            # compact summary
            if not df_out.empty:
                first_row = df_out.iloc[0]
                response_container.info(f"📊 Top row: **{first_row[0]}** — {first_row[1]}")

            # download buttons
            csv = df_out.to_csv(index=False).encode("utf-8")
            response_container.download_button("⬇ Download CSV", csv, "results.csv", "text/csv")

            buf = io.BytesIO()
            df_out.to_excel(buf, index=False, engine="openpyxl")
            response_container.download_button("⬇ Download Excel", buf.getvalue(), "results.xlsx")

            # auto chart
            if df_out.shape[1] == 2 and pd.api.types.is_numeric_dtype(df_out.iloc[:, 1]):
                chart_data = df_out.set_index(df_out.columns[0]).iloc[:, 0]
                response_container.bar_chart(chart_data)

        elif isinstance(last_answer, pd.DataFrame):
            response_container.dataframe(last_answer, use_container_width=True)
        else:
            response_container.markdown(last_answer)

    # Input box
    st.markdown("### 🔍 Enter your question")
    q_val = st.text_area("Type here...", value=st.session_state.get("question", ""), key="question_input", height=80)
    if st.button("🚀 Ask Question"):
        if not q_val.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Analyzing..."):
                answer = chat_with_delivery_data(q_val, df)
            st.session_state["question"] = q_val
            st.session_state["last_answer"] = answer
            st.experimental_rerun()
