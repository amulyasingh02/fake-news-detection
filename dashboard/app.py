import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

DATA_PATH = "C:/Users/mynam/OneDrive/Desktop/social media/data/scored_reddit_posts.csv"

st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide",
    page_icon="🧠"
)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    selected = option_menu(
        "Fake News AI Monitor",
        ["Dashboard", "Suspicious Posts", "Analytics"],
        icons=["speedometer2", "exclamation-triangle", "graph-up"],
        default_index=0
    )

# -----------------------------
# Load data
# -----------------------------

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Waiting for data from pipeline...")
    st.stop()

# -----------------------------
# Dashboard Page
# -----------------------------

if selected == "Dashboard":

    st.title("🧠 AI Fake News Detection System")

    st.markdown(
        "Real-time multimodal misinformation monitoring system"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Posts", len(df))

    fake_posts = df["prediction"].sum()
    col2.metric("Fake Predictions", int(fake_posts))

    avg_score = df["fake_probability"].mean()
    col3.metric("Average Risk Score", round(avg_score,2))

    suspicious = len(df[df["fake_probability"]>0.8])
    col4.metric("High Risk Posts", suspicious)

    st.divider()

    # Risk gauge
    st.subheader("Fake News Risk Level")

    fig = px.histogram(
        df,
        x="fake_probability",
        nbins=30,
        color="prediction",
        color_discrete_map={0:"green",1:"red"}
    )

    st.plotly_chart(fig, width="stretch")

    # Engagement vs risk
    st.subheader("Engagement vs Fake Probability")

    fig2 = px.scatter(
        df,
        x="# upvotes",
        y="fake_probability",
        size="# comments",
        hover_data=["Post text"],
        color="prediction",
        color_discrete_map={0:"green",1:"red"}
    )

    st.plotly_chart(fig2, width="stretch")

# -----------------------------
# Suspicious Posts Page
# -----------------------------

if selected == "Suspicious Posts":

    st.title("🚨 Suspicious Content Detection")

    high_risk = df.sort_values(
        "fake_probability",
        ascending=False
    ).head(20)

    for _, row in high_risk.iterrows():

        with st.container():

            risk = row["fake_probability"]

            if risk > 0.8:
                color = "red"
                label = "HIGH RISK"
            elif risk > 0.6:
                color = "orange"
                label = "MODERATE RISK"
            else:
                color = "green"
                label = "LOW RISK"

            st.markdown(
                f"""
                ### {label}
                **Fake Probability:** {risk:.2f}

                {row['Post text']}
                """
            )

            st.progress(risk)

            st.caption(
                f"Upvotes: {row['# upvotes']} | Comments: {row['# comments']}"
            )

            st.divider()

# -----------------------------
# Analytics Page
# -----------------------------

if selected == "Analytics":

    st.title("📊 AI Detection Analytics")

    col1, col2 = st.columns(2)

    # Metadata vs text score
    fig = px.scatter(
        df,
        x="text_score",
        y="meta_score",
        color="prediction",
        title="Text vs Metadata Credibility"
    )

    col1.plotly_chart(fig, width="stretch")

    # fake distribution
    fig2 = px.pie(
        df,
        names="prediction",
        title="Fake vs Real Predictions"
    )

    col2.plotly_chart(fig2, width="stretch")

    st.subheader("Top Viral Posts")

    viral = df.sort_values(
        "# upvotes",
        ascending=False
    ).head(10)

    st.dataframe(
        viral[
            [
                "Post text",
                "# upvotes",
                "# comments",
                "fake_probability"
            ]
        ]
    )