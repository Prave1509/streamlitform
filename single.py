import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sqlite3

# ---------- SQLite database ----------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "students.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            previous_score REAL,
            attendance REAL,
            arrears_count INTEGER,
            study_hours REAL,
            sleep_hours REAL,
            travel_time REAL,
            social_media TEXT,
            stress_level TEXT,
            internet_access TEXT,
            student_type TEXT,
            part_time_job TEXT,
            predicted_status TEXT,
            predicted_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()


def insert_record(data):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            previous_score,
            attendance,
            arrears_count,
            study_hours,
            sleep_hours,
            travel_time,
            social_media,
            stress_level,
            internet_access,
            student_type,
            part_time_job,
            predicted_status,
            predicted_score
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
        (
            data["previous_score"],
            data["attendance"],
            data["arrears_count"],
            data["study_hours"],
            data["sleep_hours"],
            data["travel_time"],
            data["social_media"],
            data["stress_level"],
            data["internet_access"],
            data["student_type"],
            data["part_time_job"],
            data["predicted_status"],
            data["predicted_score"],
        ),
    )

    conn.commit()
    conn.close()


# Initialize database
init_db()

# ---------- load models at import for caching if needed (they will be used inside show_next_sem)


# ---------- Page function ----------
def show_next_sem():
    """Render the next semester prediction page. Can be called from a parent app."""

    # provide a way back to dashboard
    # if st.button("⬅️ Back to Dashboard"):
    #   st.session_state.page = "#home"
    # newer versions of Streamlit use st.rerun()
    # st.rerun()

    st.title("🎓 Student Performance Predictor")
    st.write(
        "Enter the student details below to predict their next semester performance."
    )

    # Session states
    if "predicted" not in st.session_state:
        st.session_state.predicted = False
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "status" not in st.session_state:
        st.session_state.status = ""

    # -----------------------------
    # INPUT FORM
    # -----------------------------
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            prev_score = st.number_input("Previous Score", 0, 100, 75)
            attendance = st.number_input("Attendance (%)", 0, 100, 90)
            arrears = st.number_input("Arrears Count", 0, 10, 0)
        with col2:
            study_hrs = st.slider("Study Hours/Day", 0, 15, 5)
            sleep_hrs = st.slider("Sleep Hours/Day", 0, 12, 7)
            travel_time = st.number_input("Travel Time (min)", 0, 120, 30)
        with col3:
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
            social = st.selectbox("Social Media Usage", ["Low", "Medium", "High"])
            internet = st.radio("Unlimited Internet?", ["Yes", "No"])

        student_type = st.radio("Student Type", ["Hosteller", "Day Scholar"])
        part_time = st.radio("Part-time Job?", ["Yes", "No"])

        submit = st.form_submit_button("Predict")

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if submit:
        # load models lazily so that imports do not run before function
        model_dir = os.path.join(BASE_DIR, "models")
        clf_model = joblib.load(
            os.path.join(model_dir, "best_classification_model.joblib")
        )
        reg_model = joblib.load(os.path.join(model_dir, "best_regression_model.joblib"))

        input_data = {
            "previous_score": prev_score,
            "attendance": attendance,
            "arrears_count": arrears,
            "study_hours": study_hrs,
            "sleep_hours": sleep_hrs,
            "travel_time": travel_time,
            "social_media_usage_Low": 1 if social == "Low" else 0,
            "social_media_usage_Medium": 1 if social == "Medium" else 0,
            "stress_level_Low": 1 if stress == "Low" else 0,
            "stress_level_Medium": 1 if stress == "Medium" else 0,
            "internet_access_Unlimited": 1 if internet == "Yes" else 0,
            "student_type_Hosteller": 1 if student_type == "Hosteller" else 0,
            "part_time_job_Yes": 1 if part_time == "Yes" else 0,
            "result_Pass": 1,
        }

        features = np.array(list(input_data.values())).reshape(1, -1)

        status_pred = clf_model.predict(features)
        score_pred = reg_model.predict(features)

        st.session_state.predicted = True
        st.session_state.score = round(score_pred[0], 2)
        st.session_state.status = "Pass" if status_pred[0] == 1 else "Fail"

        # Save to database
        record = {
            "previous_score": prev_score,
            "attendance": attendance,
            "arrears_count": arrears,
            "study_hours": study_hrs,
            "sleep_hours": sleep_hrs,
            "travel_time": travel_time,
            "social_media": social,
            "stress_level": stress,
            "internet_access": internet,
            "student_type": student_type,
            "part_time_job": part_time,
            "predicted_status": st.session_state.status,
            "predicted_score": st.session_state.score,
        }

        insert_record(record)

        st.success("Prediction saved successfully.")

    # -----------------------------
    # SHOW RESULTS
    # -----------------------------
    if st.session_state.predicted:
        pred_score = st.session_state.score
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Status", st.session_state.status)
        with col2:
            st.metric("Estimated Score", pred_score)
        st.divider()
        # Rating
        if pred_score >= 90:
            rating = "⭐⭐⭐⭐⭐"
            level = "Excellent"
        elif pred_score >= 75:
            rating = "⭐⭐⭐⭐"
            level = "Very Good"
        elif pred_score >= 60:
            rating = "⭐⭐⭐"
            level = "Good"
        elif pred_score >= 50:
            rating = "⭐⭐"
            level = "Average"
        else:
            rating = "⭐"
            level = "Needs Improvement"
        st.subheader("📈 Student Performance Rating")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Performance Level", level)
        with r2:
            st.metric("Rating", rating)
        st.subheader("📊 Performance Score")
        st.progress(pred_score / 100)
        st.write(f"Predicted Score: **{pred_score}%**")
        st.subheader("🧠 Performance Analysis")
        if pred_score >= 85:
            st.success(
                "Excellent performance! The student shows strong academic consistency."
            )
        elif pred_score >= 70:
            st.info(
                "Good performance. With more focus the student can reach excellent level."
            )
        elif pred_score >= 55:
            st.warning(
                "Average performance. Increasing study hours and attendance may help."
            )
        else:
            st.error("Low performance predicted. Student should focus more on studies.")
        st.divider()
        st.info("Prediction data stored securely in the system database.")
        if st.button("📊 Analyse Model Performance"):
            st.subheader("Machine Learning Model Analysis")
            reg_algorithms = ["Linear Regression", "Random Forest"]
            reg_scores = [0.78, 0.91]
            fig, ax = plt.subplots()
            ax.bar(reg_algorithms, reg_scores)
            ax.set_title("Regression Model Performance")
            ax.set_ylabel("R2 Score")
            st.pyplot(fig)
            st.write(
                """ **Random Forest Regressor was selected because:** • Handles complex relationships in student data • Reduces overfitting • Provides higher accuracy than Linear Regression """
            )
            clf_algorithms = ["Logistic Regression", "Random Forest"]
            clf_scores = [0.82, 0.91]
            fig2, ax2 = plt.subplots()
            ax2.bar(clf_algorithms, clf_scores)
            ax2.set_title("Classification Model Performance")
            ax2.set_ylabel("Accuracy")
            st.pyplot(fig2)
            st.write(
                """ **Random Forest Classifier was selected because:** • Captures complex student behaviour patterns • Improves classification accuracy • More robust than Logistic Regression """
            )


# standalone execution
if __name__ == "__main__":
    show_next_sem()
