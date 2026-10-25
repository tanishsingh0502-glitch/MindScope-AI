
import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MindScope AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# LOAD FINAL MODEL
# =========================================================
model = joblib.load("linear_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# =========================================================
# TEXT CLEANING
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =========================================================
# ANALYSIS FUNCTION
# =========================================================
def analyze_text(text):

    cleaned = clean_text(text)

    if not cleaned:
        return None

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]
    scores = model.decision_function(vector)[0]

    order = np.argsort(scores)[::-1]

    top_classes = model.classes_[order]
    top_scores = scores[order]

    margin = float(top_scores[0] - top_scores[1])

    # Heuristic prediction-strength indicator
    if margin >= 1.0:
        strength = "High"
    elif margin >= 0.5:
        strength = "Medium"
    else:
        strength = "Low"

    return {
        "prediction": prediction,
        "strength": strength,
        "margin": margin,
        "top_classes": top_classes,
        "top_scores": top_scores
    }


# =========================================================
# CATEGORY INFORMATION
# =========================================================
category_info = {

    "Positive": {
        "emoji": "😊",
        "message": "The text shows positive or uplifting language.",
        "suggestion":
            "Consider noting what contributed to this positive moment "
            "or sharing it with someone you trust."
    },

    "Neutral": {
        "emoji": "😐",
        "message": "The text appears mostly neutral in emotional tone.",
        "suggestion":
            "A short reflection can help you notice how your day "
            "has been going."
    },

    "Anger": {
        "emoji": "😠",
        "message": "The text contains language associated with anger.",
        "suggestion":
            "Consider taking a short break and giving yourself time "
            "before reacting to a frustrating situation."
    },

    "Anxiety": {
        "emoji": "😟",
        "message":
            "The text contains language associated with worry or anxiety.",
        "suggestion":
            "Consider slowing down, taking a quiet break, or writing "
            "down the thoughts that are worrying you."
    },

    "Sadness": {
        "emoji": "😔",
        "message": "The text contains language associated with sadness.",
        "suggestion":
            "Consider talking with someone you trust or spending some "
            "time on a gentle activity you enjoy."
    },

    "Depression": {
        "emoji": "🌧️",
        "message":
            "The text contains patterns associated with the dataset's "
            "Depression category.",
        "suggestion":
            "If difficult feelings continue or interfere with daily life, "
            "consider talking with a trusted adult or qualified health "
            "professional."
    }
}


# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "journal" not in st.session_state:
    st.session_state.journal = []


# =========================================================
# HELPER — SAVE ANALYSIS
# =========================================================
def save_analysis(text, prediction, strength, margin, source):

    st.session_state.history.append({

        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Text": text,
        "Category": prediction,
        "Strength": strength,
        "Margin": round(margin, 3)

    })


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("🧠 MindScope AI")

    st.write(
        "Educational NLP system for dataset-defined "
        "emotion and mental-wellness text classification."
    )

    st.divider()

    st.subheader("🏆 Final Model")

    st.write("**Linear SVM — V3**")
    st.write("Test Accuracy: **97.38%**")
    st.write("Macro F1: **97.44%**")

    st.divider()

    st.subheader("🏷️ Categories")

    st.write("""
    😠 Anger  
    😟 Anxiety  
    🌧️ Depression  
    😐 Neutral  
    😊 Positive  
    😔 Sadness
    """)

    st.divider()

    st.subheader("🔐 Privacy")

    st.write(
        "Analysis history and journal entries are kept only "
        "in the current Streamlit session."
    )

    if st.button("🗑️ Clear Session Data"):

        st.session_state.history = []
        st.session_state.journal = []

        st.success("Session data cleared.")

    st.divider()

    st.caption(
        "Educational project only. "
        "MindScope AI does not provide a medical diagnosis."
    )


# =========================================================
# MAIN HEADER
# =========================================================
st.title("🧠 MindScope AI")

st.subheader(
    "Mental Wellness Classification & Analysis System"
)

st.write(
    "A machine-learning and NLP project that analyzes text "
    "using dataset-defined categories."
)

st.info(
    "ℹ️ Model Note: 97.38% accuracy was measured on a held-out "
    "test split from the same dataset distribution. Naturally written "
    "or unfamiliar text may sometimes be classified differently because "
    "language and context can vary."
)


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "🔍 Analyzer",
    "📝 Quick Journal",
    "📈 Mood Tracker",
    "📊 Model Dashboard",
    "ℹ️ About"

])


# =========================================================
# TAB 1 — ANALYZER
# =========================================================
with tab1:

    st.header("🔍 Text Analyzer")

    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Write a sentence here...",
        height=130,
        key="analyzer_text"
    )

    if st.button(
        "🧠 Analyze Text",
        type="primary",
        key="analyze_button"
    ):

        if not user_text.strip():

            st.warning("Please enter some text first.")

        else:

            result = analyze_text(user_text)

            if result is None:

                st.warning(
                    "Please enter meaningful text using words."
                )

            else:

                prediction = result["prediction"]
                strength = result["strength"]
                margin = result["margin"]
                top_classes = result["top_classes"]
                top_scores = result["top_scores"]

                info = category_info[prediction]

                st.divider()

                # -------------------------
                # RESULT
                # -------------------------
                st.subheader("📋 Analysis Result")

                st.success(
                    f"{info['emoji']} Predicted Category: "
                    f"**{prediction}**"
                )

                st.info(info["message"])

                # -------------------------
                # STRENGTH
                # -------------------------
                st.subheader("🎯 Prediction Strength")

                if strength == "High":

                    st.success("🟢 HIGH")

                elif strength == "Medium":

                    st.warning("🟡 MEDIUM")

                else:

                    st.error("🔴 LOW")

                st.caption(
                    f"SVM decision margin: {margin:.3f}"
                )

                if strength == "Low":

                    st.warning(
                        "The leading model categories were relatively "
                        "close. Interpret this result cautiously."
                    )

                # -------------------------
                # TOP 3
                # -------------------------
                st.subheader("🥇 Top 3 Model Matches")

                top3_df = pd.DataFrame({

                    "Rank": [
                        "1st",
                        "2nd",
                        "3rd"
                    ],

                    "Category":
                        top_classes[:3],

                    "Decision Score":
                        np.round(top_scores[:3], 3)

                })

                st.dataframe(
                    top3_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "SVM decision scores represent relative model "
                    "preference and are not probabilities."
                )

                # -------------------------
                # SUGGESTION
                # -------------------------
                st.subheader(
                    "💡 General Wellness Suggestion"
                )

                st.write(info["suggestion"])

                action_suggestions = {

                    "Positive": [
                        "📝 Write down what made today feel good.",
                        "🤝 Share the positive moment with someone you trust.",
                        "🎯 Use the good momentum to complete a small goal."
                    ],

                    "Neutral": [
                        "📝 Take a minute to reflect on how your day is going.",
                        "🚶 Take a short break or walk if you have been sitting for long.",
                        "🎯 Choose one small useful task for the rest of the day."
                    ],

                    "Anger": [
                        "⏸️ Give yourself a short pause before reacting.",
                        "🌿 Move to a quieter place if possible.",
                        "📝 Write down what specifically frustrated you.",
                        "🤝 Talk it through calmly with someone you trust."
                    ],

                    "Anxiety": [
                        "🌬️ Try a few slow, comfortable breaths.",
                        "📝 Write down the specific thing that is worrying you.",
                        "🎯 Focus on one small thing you can handle right now.",
                        "🤝 Consider talking to a trusted person for support."
                    ],

                    "Sadness": [
                        "🤝 Consider spending some time with someone you trust.",
                        "🎵 Do a gentle activity you usually enjoy.",
                        "📝 Write down what has been on your mind.",
                        "🌤️ Give yourself a small, manageable task instead of trying to fix everything at once."
                    ],

                    "Depression": [
                        "🤝 Consider telling a trusted adult how you have been feeling.",
                        "🌤️ Keep expectations small and focus on manageable daily activities.",
                        "📝 Writing down your thoughts may help you organize what you are feeling.",
                        "🧑‍⚕️ If difficult feelings persist or interfere with daily life, consider speaking with a qualified health professional."
                    ]
                }

                st.markdown("#### ✅ Suggested Actions")

                for action in action_suggestions[prediction]:
                    st.write(action)

                st.caption(
                    "These are general wellness ideas for educational purposes, "
                    "not medical advice or treatment."
                )

                # -------------------------
                # SAVE HISTORY
                # -------------------------
                save_analysis(
                    user_text,
                    prediction,
                    strength,
                    margin,
                    "Analyzer"
                )


# =========================================================
# TAB 2 — QUICK JOURNAL
# =========================================================
with tab2:

    st.header("📝 Quick Journal")

    st.write(
        "Write a short reflection about your day. "
        "The entry can be analyzed and added to the "
        "current session tracker."
    )

    journal_text = st.text_area(
        "Journal entry:",
        placeholder=(
            "Example: Today was busy but I enjoyed "
            "spending time with my friends..."
        ),
        height=180,
        key="journal_text"
    )

    if st.button(
        "📖 Save & Analyze Entry",
        key="journal_button"
    ):

        if not journal_text.strip():

            st.warning(
                "Write something in the journal first."
            )

        else:

            result = analyze_text(journal_text)

            if result is None:

                st.warning(
                    "Please enter meaningful text using words."
                )

            else:

                prediction = result["prediction"]
                strength = result["strength"]
                margin = result["margin"]

                info = category_info[prediction]

                entry = {

                    "Time":
                        datetime.now().strftime("%H:%M:%S"),

                    "Entry":
                        journal_text,

                    "Category":
                        prediction,

                    "Strength":
                        strength

                }

                st.session_state.journal.append(entry)

                save_analysis(
                    journal_text,
                    prediction,
                    strength,
                    margin,
                    "Journal"
                )

                st.success(
                    "Journal entry added to this session."
                )

                st.write(
                    f"### {info['emoji']} "
                    f"Detected Category: {prediction}"
                )

                st.write(info["message"])

                st.info(
                    "💡 " + info["suggestion"]
                )

    # Journal list
    if len(st.session_state.journal) > 0:

        st.divider()

        st.subheader("📚 Current Session Journal")

        journal_df = pd.DataFrame(
            st.session_state.journal
        )

        st.dataframe(
            journal_df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Journal entries are session-only and are not "
        "intended to form a medical record."
    )


# =========================================================
# TAB 3 — MOOD / CATEGORY TRACKER
# =========================================================
with tab3:

    st.header("📈 Session Mood Tracker")

    st.write(
        "A summary of categories predicted during "
        "the current application session."
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No session analyses yet. Use the Analyzer "
            "or Quick Journal first."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # -------------------------
        # METRICS
        # -------------------------
        total_analyses = len(history_df)

        low_predictions = (
            history_df["Strength"] == "Low"
        ).sum()

        most_common = (
            history_df["Category"]
            .value_counts()
            .idxmax()
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Total Analyses",
                total_analyses
            )

        with col2:

            st.metric(
                "Most Frequent Category",
                most_common
            )

        with col3:

            st.metric(
                "Low-Strength Results",
                int(low_predictions)
            )

        # -------------------------
        # CATEGORY DISTRIBUTION
        # -------------------------
        st.subheader(
            "📊 Category Distribution"
        )

        category_counts = (
            history_df["Category"]
            .value_counts()
            .rename_axis("Category")
            .to_frame("Count")
        )

        st.bar_chart(category_counts)

        # -------------------------
        # HISTORY
        # -------------------------
        st.subheader(
            "🕒 Session History"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------
        # DOWNLOAD
        # -------------------------
        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Session Report (CSV)",
            data=csv_data,
            file_name="mindscope_session_report.csv",
            mime="text/csv"
        )

        st.caption(
            "The downloaded report contains only the "
            "analyses performed in this current session."
        )


# =========================================================
# TAB 4 — MODEL DASHBOARD
# =========================================================
with tab4:

    st.header(
        "📊 Machine Learning Model Comparison"
    )

    results = pd.DataFrame({

        "Model": [
            "Linear SVM (V3)",
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "KNN"
        ],

        "Accuracy": [
            0.9738,
            0.9506,
            0.9251,
            0.9222,
            0.8623,
            0.2455
        ],

        "Macro F1": [
            0.9744,
            0.9520,
            0.9272,
            0.9250,
            0.8657,
            0.1664
        ]

    })

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------
    # ACCURACY CHART
    # -------------------------
    st.subheader(
        "📈 Accuracy Comparison"
    )

    st.bar_chart(
        results[
            ["Model", "Accuracy"]
        ].set_index("Model")
    )

    # -------------------------
    # F1 CHART
    # -------------------------
    st.subheader(
        "📈 Macro F1 Comparison"
    )

    st.bar_chart(
        results[
            ["Model", "Macro F1"]
        ].set_index("Model")
    )

    st.success(
        "🏆 Final Linear SVM V3 achieved "
        "97.38% test accuracy and "
        "97.44% Macro F1."
    )

    st.info(
        "V3 combines word-level and character-level "
        "TF-IDF features before Linear SVM classification."
    )

    st.warning(
        "High test-set performance does not guarantee "
        "perfect predictions on naturally written text "
        "outside the dataset distribution."
    )


# =========================================================
# TAB 5 — ABOUT PROJECT
# =========================================================
with tab5:

    st.header("ℹ️ About MindScope AI")

    st.write(
        """
        **MindScope AI** is an educational Natural Language
        Processing and Machine Learning project.

        It classifies text into six categories defined by
        the training dataset and demonstrates a complete
        machine-learning workflow from data preprocessing
        to an interactive web application.
        """
    )

    st.subheader("⚙️ Project Workflow")

    st.markdown(
        """
        **1. Dataset collection**

        ↓

        **2. Data exploration and cleaning**

        ↓

        **3. Train-test split**

        ↓

        **4. TF-IDF feature extraction**

        ↓

        **5. Six machine-learning models evaluated**

        ↓

        **6. Linear SVM selected**

        ↓

        **7. Word + character TF-IDF V3 improvement**

        ↓

        **8. Streamlit web application**
        """
    )

    st.subheader("🤖 Models Compared")

    st.write(
        """
        • Logistic Regression  
        • Multinomial Naive Bayes  
        • Linear SVM  
        • Random Forest  
        • Decision Tree  
        • K-Nearest Neighbors
        """
    )

    st.subheader("🏆 Final Model")

    st.write(
        "**Linear SVM V3** using a combination of "
        "word-level and character-level TF-IDF features."
    )

    st.write(
        "**Test Accuracy:** 97.38%"
    )

    st.write(
        "**Macro F1 Score:** 97.44%"
    )

    st.subheader("🎯 Prediction Strength")

    st.write(
        "Linear SVM does not naturally provide calibrated "
        "probabilities. MindScope AI therefore uses the "
        "margin between the two highest decision scores as "
        "a heuristic indication of prediction strength."
    )

    st.subheader("⚠️ Project Limitations")

    st.write(
        """
        • Predictions depend on patterns present in the dataset.

        • Naturally written text may differ from the training
        data and can therefore be misclassified.

        • High SVM decision margin does not guarantee that a
        prediction is semantically correct.

        • Decision scores are not probabilities.

        • Prediction strength is a heuristic indicator.

        • Dataset-defined labels should not be interpreted as
        clinical diagnoses.

        • The application is intended for educational
        demonstration only.
        """
    )

    st.subheader("🔐 Privacy Design")

    st.write(
        "Journal entries and analysis history are stored "
        "only in Streamlit session state during the current "
        "running session. MindScope AI does not intentionally "
        "create a permanent personal mental-health record."
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "🧠 MindScope AI | Educational ML & NLP Project | "
    "Not intended for medical diagnosis."
)
