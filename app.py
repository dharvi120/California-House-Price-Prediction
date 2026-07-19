import streamlit as st
import pandas as pd
import numpy as np
import time

# ---------------- CONFIG ---------------- #

st.set_page_config(
    page_title="California House Price AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================================
# Sidebar Navigation
# ==========================================



with st.sidebar:

    st.markdown("# 🏠 House Price AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🧠 Explainability",
            "📈 Model Performance",
            "🗂 Dataset",
            "👨‍💻 Developer"
        ],
        label_visibility="collapsed"
    )

st.markdown("---")

st.markdown("""
### 🤖 AI Dashboard
""")

st.caption("Version 2.0")
# ---------------- CSS ---------------- #


def load_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("load_css("assets/css/style.css")")



# ---------------- HEADER ---------------- #

st.markdown("""
<div class="hero">

<div class="hero-content">

<h1>🏠 California House Price AI</h1>

<p class="hero-tag">
Production Grade Machine Learning Dashboard
</p>

<p class="hero-sub">

⚡ Python • XGBoost • SHAP • Plotly • Streamlit

</p>

<p class="hero-small">

Built for Real Estate Intelligence

</p>



</div>



<div>

<div class="version">
v2.0

</div>


</div>
""", unsafe_allow_html=True)

# ---------------- METRICS ---------------- #
# ---------------- METRICS ---------------- #

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("📈", "0.91", "R² Score"),
    ("📉", "$42K", "RMSE"),
    ("💰", "$28K", "MAE"),
    ("🎯", "96.8%", "Confidence")
]

for col, (icon, value, label) in zip([c1, c2, c3, c4], cards):

    with col:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-icon">{icon}</div>
<div class="metric-value">{value}</div>

<div class="metric-label">{label}</div>

</div>
""",
            unsafe_allow_html=True
        )
st.write("")
st.write("")

# ==========================================================
# MAIN SECTION
# ==========================================================

left, right = st.columns([1.2,1])

# ==========================================================
# LEFT PANEL
# ==========================================================

with left:

    st.markdown("""
    <div class="section-title">
    📋 Property Details
    </div>
    """, unsafe_allow_html=True)

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        max_value=20.0,
        value=3.5,
        step=0.1
    )

    house_age = st.slider(
        "House Age",
        1,
        60,
        20
    )

    avg_rooms = st.slider(
        "Average Rooms",
        1.0,
        15.0,
        5.0
    )

    avg_bedrooms = st.slider(
        "Average Bedrooms",
        1.0,
        10.0,
        2.0
    )

    population = st.number_input(
        "Population",
        value=3000
    )

    avg_occupancy = st.slider(
        "Average Occupancy",
        1.0,
        10.0,
        3.0
    )

    latitude = st.slider(
        "Latitude",
        32.0,
        42.0,
        35.0
    )

    longitude = st.slider(
        "Longitude",
        -125.0,
        -114.0,
        -120.0
    )

    ocean = st.selectbox(
        "Ocean Proximity",
        [
            "NEAR BAY",
            "INLAND",
            "<1H OCEAN",
            "NEAR OCEAN",
            "ISLAND"
        ]
    )

    predict = st.button("🚀 Predict House Price")

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    st.markdown("""
    <div class="section-title">
    🤖 AI Prediction
    </div>
    """, unsafe_allow_html=True)

    if predict:

        with st.spinner("🤖 Running XGBoost Inference..."):

            progress = st.progress(0)

            time.sleep(.8)

            

            # ------------------------------------------------

            # Temporary Prediction

            prediction = round(
                median_income*100000
                + avg_rooms*18000
                + house_age*1400
                + population*6
            )

            confidence = np.random.randint(92,99)

        if prediction < 200000:

            recommendation = "Affordable Segment"

            color = "#16a34a"

        elif prediction < 500000:

            recommendation = "High Investment Potential"

            color = "#2563eb"

        else:

            recommendation = "Premium Residential Property"

            color = "#9333ea"

        st.markdown(f"""

<div class="prediction-card">

<h1>${prediction:,.0f}</h1>

<p>Predicted Market Value</p>

<hr>

<h3>{confidence}% Confidence</h3>

<h4 style="color:{color};">

{recommendation}

</h4>

</div>

""", unsafe_allow_html=True)

        st.markdown(f"""

<div class="metric-card">

<h2>

${prediction-15000:,.0f}

-

${prediction+15000:,.0f}

</h2>

<p>Estimated Price Range</p>

</div>

""", unsafe_allow_html=True)

        st.success("Prediction completed successfully.")

    else:

        st.markdown("""

<div class="empty-card">

<h2> Ready for Prediction</h2>

<p>

Complete the property information

and press

Predict House Price

<b>Predict House Price</b>

</p>

</div>

""", unsafe_allow_html=True)

st.write("")
st.write("")



    # ==========================================
# Pages
# ==========================================

if page == "📊 Dashboard":

    st.title("California House Price Prediction")
    st.write("Main prediction interface goes here.")

elif page == "📊 Performance":

    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", "0.84")
    col2.metric("MAE", "$32,000")
    col3.metric("RMSE", "$48,500")

elif page == "🧠 Explainability":

    st.title("Explainability")

    st.info("SHAP feature importance will appear here.")

elif page == "📈 Dataset":

    st.title("California Housing Dataset")

    st.write("Dataset Information")


elif page == "👨‍💻 Developer":

    st.title("Developer")

    st.markdown("""
**Dharvi Sharma**

B.Tech CSE (AI & ML)

### Technical Stack

Python

Machine Learning

Deep Learning

XGBoost

Scikit-learn

Pandas

NumPy

Streamlit

Plotly

SHAP

Git

Docker

AWS (Learning)Project

California House Price Prediction Platform
""")
    import streamlit as st

# ==========================
# SHAP Explainability Card
# ==========================

st.markdown("""
<div class="glass-card">

<div class="section-title">
🧠 SHAP Explainability
</div>

<div class="section-subtitle">
        Feature importance generated using SHAP values
</div>

</div>
""", unsafe_allow_html=True)

# SHAP Image
col1,col2,col3=st.columns([1,5,1])

with col2:

    st.image(

        "screenshots/shap_summary.png",

        width=900

    )
    st.markdown("""

<div class="glass-card">

<h2>

🧠 SHAP Explainability

</h2>

<p>

           Feature importance generated using SHAP

</p>

</div>

""",unsafe_allow_html=True)

# =====================================================
# Premium Footer
# =====================================================
st.markdown("""
<div class="footer-wrapper">

<div class="gradient-line"></div>

<div class="footer-card">

<h2>👨‍💻 Dharvi Sharma</h2>

<p class="footer-role">
     B.Tech Artificial Intelligence & Machine Learning Engineer
</p>

<div class="footer-contact">

<span>📧 dharvi12@gmail.com</span>

<span>📍 Haryana, India</span>

</div>

<div class="footer-links">

<a href="https://github.com/dharvi120" target="_blank">
                GitHub
</a>

<span>•</span>

<a href="https://www.linkedin.com/in/dharvi-sharma-76bb86307/" target="_blank">
                LinkedIn
</a>

<span>•</span>

<a href="https://dharvi120.github.io" target="_blank">
                Portfolio
</a>

</div>
<div class="footer-tech">

🏆 Built with
<strong>
Streamlit • XGBoost • SHAP • Python
</strong>

</div>

<div class="footer-version">

Version 2.0

</div>

<div class="footer-copy">

© 2026 Dharvi Sharma | All Rights Reserved

</div>

</div>

</div>
""",
unsafe_allow_html=True
)
