import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Streamlit Portfolio & Data Viz", page_icon="📊", layout="wide")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "📄 Bio", "📊 EDA Gallery", "📈 Dashboard", "🧭 Future Work"])

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    # expects the Kaggle diabetes.csv to be next to this file
    df = pd.read_csv("diabetes.csv")
    # Make categorical Outcome label for convenience
    if "Outcome" in df.columns:
        df["OutcomeLabel"] = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}).astype("category")
    return df

try:
    df = load_data()
except Exception as e:
    df = None

# Common helpers
def numeric_cols(d):
    return d.select_dtypes(include=["number"]).columns.tolist() if d is not None else []

# -----------------------------
# HOME
# -----------------------------
if page == "Home":
    st.title("📊 Streamlit Portfolio & Mini Analytics Product")
    st.write("Use the left sidebar to navigate between pages.")
    st.markdown("---")
    st.subheader("What’s inside")
    st.markdown(
        """
        - **📄 Bio**: Short professional summary and highlights  
        - **📊 EDA Gallery**: Four distinct charts + “how to read this chart” + observations  
        - **📈 Dashboard**: Filters, KPIs, and linked visuals reacting together  
        - **🧭 Future Work**: Next steps and brief reflection
        """
    )
    st.caption(f"Last opened: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -----------------------------
# BIO
# -----------------------------
elif page == "📄 Bio":
    st.title("📄 Professional Bio")
    left, right = st.columns([1, 2], gap="large")

    with left:
        # If you later add coder_avatar.png to an assets/ folder, change this path to "assets/coder_avatar.png"
        st.image(
            "assets/coder_avatar.png", caption="Alt: avatar illustration of a coder"
        )

    with right:
        st.subheader("About Me")
        st.write(
            "Hi, I’m Umulbanin Gulzar. I’m a Computer Science major who enjoys working with data, "
            "building clean visual dashboards, and learning how to spot patterns through code. "
            "I like turning messy data into something clear and easy to understand. "
            "I’ve worked with tools like Python, Streamlit, Pandas, and Tableau to build "
            "interactive dashboards and small apps. I also enjoy learning topics like "
            "algorithms, data visualization, and beginner-level machine learning."
        )

        st.subheader("Highlights")
        st.markdown(
            """
            - Coursework in Data Visualization, Algorithms, and basic Machine Learning  
            - Strong with Python (Pandas, Plotly, Streamlit)  
            - Experience building EDA dashboards and interactive charts  
            - Comfortable with GitHub and version control  
            - Interested in learning more about AI and predictive modeling  
            """
        )

        st.subheader("Visualization Philosophy")
        st.write(
            "I believe data visualizations should be clear, simple, and easy for everyone to understand. "
            "I try to use readable colors, good labels, and layouts that don’t overwhelm the viewer. "
            "My goal is to explain what the data shows without over-complicating it."
        )

# -----------------------------
# EDA GALLERY
# -----------------------------
elif page == "📊 EDA Gallery":
    st.title("📊 EDA Charts Gallery")

    if df is None or df.empty:
        st.warning("I can’t find `diabetes.csv`. Place it next to `app.py` and reload.")
        st.stop()

    st.caption(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    ncols = numeric_cols(df)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]

    # 1) Histogram
    st.markdown("### 1) Distribution — Histogram")
    if ncols:
        num_col = st.selectbox("Choose a numeric column:", ncols, key="hist_num")
        fig = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of {num_col}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**How to read this chart:**")
        st.markdown(
            "- X-axis: values of the selected numeric column  \n"
            "- Y-axis: count of records in each bin  \n"
            "- Look for skew (long tail), peaks, or gaps"
        )
        st.markdown("**Your observations (3–6 bullets):**")
        st.write("- e.g., Distribution is right-skewed; most values cluster between X and Y.")

    st.markdown("---")

    # 2) Scatter
    st.markdown("### 2) Relationship — Scatter Plot")
    if len(ncols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            xcol = st.selectbox("X-axis:", ncols, index=0, key="scatter_x")
        with col2:
            ycol = st.selectbox("Y-axis:", ncols, index=1, key="scatter_y")
        color_opt = st.selectbox("Color (categorical):", ["(none)", "OutcomeLabel"] + cat_cols, key="scatter_color")
        color_arg = None if color_opt == "(none)" else color_opt
        fig = px.scatter(df, x=xcol, y=ycol, color=color_arg, title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**How to read this chart:**")
        st.markdown(
            "- Each point is a record; its position shows the values on X and Y  \n"
            "- Color groups categories (e.g., Outcome)  \n"
            "- Look for positive/negative trends, clusters, or outliers"
        )
        st.markdown("**Your observations (3–6 bullets):**")
        st.write("- e.g., Outcome=1 tends to appear at higher glucose values.")

    st.markdown("---")

    # 3) Bar (mean by category)
    st.markdown("### 3) Comparison — Bar Chart")
    if "OutcomeLabel" in df.columns and ncols:
        val = st.selectbox("Numeric to aggregate:", ncols, key="bar_val")
        agg = st.selectbox("Aggregation:", ["mean", "median", "sum", "count"], key="bar_agg")
        grouped = df.groupby("OutcomeLabel")[val].agg(agg).reset_index()
        fig = px.bar(grouped, x="OutcomeLabel", y=val, title=f"{agg.title()} of {val} by Outcome")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**How to read this chart:**")
        st.markdown(
            "- X-axis: category (Outcome)  \n"
            "- Y-axis: aggregated value  \n"
            "- Compare bar heights to see group differences"
        )
        st.markdown("**Your observations (3–6 bullets):**")
        st.write("- e.g., People with diabetes show higher mean glucose than those without.")
    else:
        st.info("I’ll use Outcome as the default category once the column exists.")

    st.markdown("---")

    # 4) Box Plot
    st.markdown("### 4) Spread — Box Plot")
    if "OutcomeLabel" in df.columns and ncols:
        val2 = st.selectbox("Numeric for box plot:", ncols, key="box_val")
        fig = px.box(df, x="OutcomeLabel", y=val2, points="outliers", title=f"Distribution of {val2} by Outcome")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**How to read this chart:**")
        st.markdown(
            "- Median (line), IQR (box), and potential outliers (points)  \n"
            "- Compare medians and spread across groups  \n"
            "- Wide boxes/outliers indicate higher variability"
        )
        st.markdown("**Your observations (3–6 bullets):**")
        st.write("- e.g., BMI shows higher median and wider spread in Outcome=1.")

# -----------------------------
# DASHBOARD
# -----------------------------
elif page == "📈 Dashboard":
    st.title("📈 Interactive Dashboard — Diabetes Dataset")
    st.markdown(
        "**Source:** [Kaggle – Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)  "
        f"· **Last refreshed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if df is None or df.empty:
        st.warning("I can’t find `diabetes.csv`. Place it next to `app.py` and reload.")
        st.stop()

    ncols = numeric_cols(df)
    if "OutcomeLabel" not in df.columns:
        st.warning("Outcome column not found. Make sure you're using the Kaggle diabetes dataset.")
        st.stop()

    # Sidebar filters
    with st.sidebar:
        st.header("Dashboard Filters")
        outcome_sel = st.multiselect("Outcome", sorted(df["OutcomeLabel"].dropna().unique()), default=list(sorted(df["OutcomeLabel"].dropna().unique())))
        range_col = st.selectbox("Numeric range column", ["(none)"] + ncols)
        if range_col != "(none)":
            min_v, max_v = float(df[range_col].min()), float(df[range_col].max())
            sel_min, sel_max = st.slider("Select range:", min_value=min_v, max_value=max_v, value=(min_v, max_v))
        else:
            sel_min, sel_max = None, None

    # Apply filters
    filtered = df.copy()
    if outcome_sel:
        filtered = filtered[filtered["OutcomeLabel"].isin(outcome_sel)]
    if range_col != "(none)" and sel_min is not None:
        filtered = filtered[(filtered[range_col] >= sel_min) & (filtered[range_col] <= sel_max)]

    # KPIs
    st.subheader("Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Rows (after filters)", len(filtered))
    with k2:
        st.metric("Mean Glucose", f"{filtered['Glucose'].mean():.1f}" if 'Glucose' in filtered.columns else "—")
    with k3:
        st.metric("Mean BMI", f"{filtered['BMI'].mean():.1f}" if 'BMI' in filtered.columns else "—")
    with k4:
        st.metric("Diabetes Rate", f"{(filtered['Outcome'].mean()*100):.1f}%" if 'Outcome' in filtered.columns else "—")

    st.markdown("---")
    st.subheader("Linked Visuals")

    # Visual 1: Bar (mean Glucose by Outcome)
    if "OutcomeLabel" in filtered.columns and "Glucose" in filtered.columns:
        g = filtered.groupby("OutcomeLabel")["Glucose"].mean().reset_index()
        fig1 = px.bar(g, x="OutcomeLabel", y="Glucose", title="Mean Glucose by Outcome")
        st.plotly_chart(fig1, use_container_width=True)

    # Visual 2: Scatter (BMI vs Glucose colored by Outcome)
    if set(["BMI", "Glucose", "OutcomeLabel"]).issubset(filtered.columns):
        fig2 = px.scatter(filtered, x="BMI", y="Glucose", color="OutcomeLabel", title="BMI vs Glucose by Outcome")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Insights (3–6 bullets)")
    st.markdown(
        """
        - People with **Outcome=1** generally show higher **Glucose** on average.  
        - The **BMI vs Glucose** scatter suggests clusters where diabetes is more prevalent at higher values.  
        - Use the numeric range filter to see how patterns change across value bands.  
        - **Limitations:** This is a small, anonymized dataset; avoid causal claims.
        """
    )

# -----------------------------
# FUTURE WORK
# -----------------------------
elif page == "🧭 Future Work":
    st.title("🧭 Future Work & Reflection")

    st.subheader("Next Steps (3–5)")
    st.markdown(
        """
        - Add model-based predictions (logistic regression) and show probability distributions.  
        - A/B test alternative dashboard layouts for clarity and minimal cognitive load.  
        - Do an accessibility audit (contrast, alt text, chart titles, keyboard navigation).  
        - Create simple feature engineering (BMI bins, age groups) to explore patterns.  
        - Automate refresh notes and add a short provenance/ethics section inside the app.  
        """
    )

    st.subheader("Reflection")

    st.markdown("**What changed from my first paper prototype to the final version?**")
    st.write(
        "In the beginning, my paper prototype was very basic. I only planned a few simple charts "
        "and didn’t really know how the dashboard would look. As I built the app in Streamlit, the layout "
        "changed a lot. I added better spacing, cleaner sections, and more interactive elements. I also "
        "realized some ideas that looked good on paper didn’t work well on the screen, so I adjusted the "
        "design as I went."
    )

    st.markdown("**Which charts stayed the same and which ones changed?**")
    st.write(
        "Some charts, like the histogram and scatter plot, stayed the same because they matched the questions "
        "I wanted to answer. Other charts changed—especially the bar chart and box plot. When I saw the actual "
        "diabetes data, I made changes so the charts would make more sense and show real patterns, not just fill "
        "space. I also added explanations and 'how to read this chart' sections to make everything clearer."
    )

    st.markdown("**How did feedback or the assignment instructions change my choices?**")
    st.write(
        "The assignment requirements helped me stay organized. They pushed me to add accessibility notes, make the "
        "charts clearer, and include KPIs on the dashboard. I also realized I needed more filters to make the "
        "dashboard useful. Overall, the instructions helped me make everything cleaner and more professional."
    )

    st.markdown("**What was the hardest part of turning the idea into the real app?**")
    st.write(
        "The most difficult part was connecting everything — the filters, the KPIs, and the linked visuals. "
        "Sometimes things broke, and I had to fix errors or rethink how to build something. Once I got the structure "
        "right, things became easier and the app started to feel more complete."
    )

    st.markdown("**What would I improve if I had more time?**")
    st.write(
        "If I had more time, I would add prediction models, like a logistic regression that shows a probability of "
        "diabetes. I would also improve the color themes and make the app even more interactive. Finally, I’d work on "
        "the storytelling part—making the insights clearer and more helpful for someone who is new to the dataset."
    )

