🚚 Delivery Data Chatbot

Live Demo: https://delivery-chatbot-scwqochc7mtnkykkayatxn.streamlit.app/

🔍 Project Overview

Delivery Data Chatbot is an interactive, AI-driven web app built on Streamlit that allows users (analysts, operations managers, stakeholders) to query delivery performance data using natural language.
Using your delivery dataset (CSV), you can ask questions like:

“Which delivery person is the fastest on average?”

“What is the average delivery time per city?”

“Does weather or traffic affect delivery speed?”

“Which areas have the highest delays?”, etc.

The app responds in real time, showing clear results in tables, stats, and visual charts.

✅ Key Features
Feature	What It Does
Natural Language Q&A	Ask plain English questions and get data-driven responses.
Performance & Efficiency	See fastest personnel, vehicle efficiency, and multi-order effects.
Environmental Analysis	Investigate traffic, weather, festival impact on delivery times.
Geospatial Insights	Highlight areas with delays and check distance-time correlation.
Downloadable Results	Export tables as CSV or Excel for offline analysis.
Interactive Visuals	Auto bar charts for comparisons (e.g. by city or traffic).
Sample Questions Sidebar	Quickly click categorized questions to explore the dataset.
📊 Stakeholder Question Types (Supported)

Delivery Performance

Which delivery person is the fastest on average?

What is the average delivery time per city?

How do multiple deliveries affect delivery time?

Which vehicle type is most efficient for deliveries?

Customer & Order Insights

What types of orders (snack, meal, drinks) take the longest?

Does order time (morning / evening) affect speed?

Are deliveries slower during festival periods?

Environmental & External Factors

How does traffic density impact delivery time?

Do weather conditions affect delivery speed?

How do restaurant vs. delivery locations influence time?

Delivery Personnel Metrics

Who has the highest ratings and fastest deliveries?

Does delivery person’s age correlate with speed or rating?

Does vehicle condition impact delivery time?

Geospatial Insights

Which areas have the highest delays?

What is the correlation between distance and delivery time?

🛠 Tech Stack & Architecture

Streamlit — for rapid web UI

Pandas / NumPy — for data processing & statistics

Python — core logic

Haversine formula — to compute distances (km) between lat/long

Session State — retains last question / answer

Interactive charts — st.bar_chart, DataFrame display

Download buttons — export CSV/Excel results

🚀 How to Use

Open the app via the link above or host it yourself.

On the left sidebar, click on categorized example questions (e.g. in “📊 Delivery Performance”) or type your own.

The AI Response appears above the input box — results displayed in bold text, tables, or charts.

Download your results (CSV / Excel) for further analysis.
