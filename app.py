import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
import hashlib
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Enterprise AutoML Analytics",
    layout="wide"
)

# ---------------- DATABASE CONNECTION ----------------
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",   # change if needed
        database="analytics"
    )

# ---------------- LOGIN FUNCTION ----------------
def login_user(username, password):

    hashed = hashlib.sha256(password.encode()).hexdigest()

    db = connect_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM users WHERE username=%s AND password_hash=%s",
        (username, hashed)
    )

    user = cursor.fetchone()

    db.close()

    return user


# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user = None


# ---------------- LOGIN PAGE ----------------
if st.session_state.user is None:

    st.title("🔐 Login to Analytics Platform")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = login_user(username, password)

        if user:
            st.session_state.user = user
            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid username or password")

    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Analytics Platform")

menu = st.sidebar.radio(
    "Navigation",
    ["Upload & Predict", "Prediction Table", "Dashboard", "Download Data", "Logout"]
)

# ---------------- UPLOAD & PREDICT ----------------
if menu == "Upload & Predict":

    st.title("🤖 AutoML Prediction")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:

        df = pd.read_csv(file)

        # limit large data for speed
        if len(df) > 2000:
            df = df.sample(2000, random_state=42)
            st.info("Large dataset detected. Using sample for faster processing.")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Run Prediction"):

            with st.spinner("Training model..."):

                X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
                y = df[target]

                # choose model automatically
                if y.nunique() <= 20:
                    model = RandomForestClassifier(
                        n_estimators=10,
                        max_depth=6,
                        n_jobs=-1,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=10,
                        max_depth=6,
                        n_jobs=-1,
                        random_state=42
                    )

                model.fit(X, y)

                preds = model.predict(X)

                confidence = np.ones(len(preds))

                result = pd.DataFrame({
                    "prediction": preds.astype(str),
                    "confidence": confidence
                })

                db = connect_db()
                cursor = db.cursor()

                records = [
                    (r["prediction"], float(r["confidence"]), st.session_state.user["username"])
                    for _, r in result.iterrows()
                ]

                cursor.executemany(
                    "INSERT INTO predictions (prediction,confidence,created_by) VALUES (%s,%s,%s)",
                    records
                )

                db.commit()
                db.close()

            st.success("Prediction completed and stored in database")


# ---------------- TABLE ----------------
elif menu == "Prediction Table":

    st.title("📄 Prediction Records")

    db = connect_db()

    df = pd.read_sql("SELECT * FROM predictions", db)

    db.close()

    st.dataframe(df, use_container_width=True)


# ---------------- DASHBOARD ----------------
elif menu == "Dashboard":

    st.title("📈 Analytics Dashboard")

    db = connect_db()

    df = pd.read_sql("SELECT * FROM predictions", db)

    db.close()

    if df.empty:
        st.warning("No predictions found.")
        st.stop()

    # KPI CARDS
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Predictions", len(df))
    c2.metric("Average Confidence", round(df.confidence.mean(), 2))
    c3.metric("Unique Predictions", df.prediction.nunique())
    c4.metric("High Confidence %",
              f"{round((df.confidence >= 0.8).mean()*100,1)}%")

    st.divider()

    col1, col2 = st.columns(2)

    # DONUT CHART
    fig1 = px.pie(
        df,
        names="prediction",
        hole=0.6,
        title="Prediction Distribution"
    )

    col1.plotly_chart(fig1, use_container_width=True)

    # BAR CHART
    fig2 = px.bar(
        df,
        x="prediction",
        title="Prediction Frequency",
        color="prediction"
    )

    col2.plotly_chart(fig2, use_container_width=True)

    # LINE CHART
    fig3 = px.line(
        df.sort_values("created_at"),
        x="created_at",
        y="confidence",
        title="Confidence Trend Over Time"
    )

    st.plotly_chart(fig3, use_container_width=True)


# ---------------- DOWNLOAD ----------------
elif menu == "Download Data":

    st.title("⬇ Download Predictions")

    db = connect_db()

    df = pd.read_sql("SELECT * FROM predictions", db)

    db.close()

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )


# ---------------- LOGOUT ----------------
elif menu == "Logout":

    st.session_state.clear()

    st.success("Logged out")

    st.rerun()
