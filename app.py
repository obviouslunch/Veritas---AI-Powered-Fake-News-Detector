
import streamlit as st
import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# -------------------------------
# 1. LOAD THE SAVED MODEL
# -------------------------------

with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# -------------------------------
# 2. PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background: #101010;
    color: #f2f0eb;
    font-family: 'Space Grotesk', sans-serif;
}

.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stHeader"] {
    height: 0rem;
    background: #101010;
}
[data-testid="stSidebar"] {
    background: #151515;
    border-right: 1px solid #353330;
}

h1, h2, h3, p, label {
    color: #f2f0eb;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stTextInput input, .stTextArea textarea {
    background: #191919 !important;
    color: #f2f0eb !important;
    border: 1px solid #41403d !important;
    border-radius: 5px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #ff453a !important;
    box-shadow: 0 0 0 1px #ff453a !important;
}

.stButton button, .stDownloadButton button {
    background: #ff453a !important;
    color: white !important;
    border: 1px solid #ff453a !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stMetric"] {
    background: #191919;
    border: 1px solid #353330;
    padding: 18px;
    border-radius: 5px;
}

[data-testid="stMetricLabel"] {
    color: #92918d !important;
    font-family: 'IBM Plex Mono', monospace;
}

[data-testid="stMetricValue"] {
    color: #f2f0eb !important;
}

.stProgress > div > div {
    background: #ff453a !important;
}

hr {
    border-color: #353330 !important;
}

.veritas-header {
    border-top: 3px solid #ff453a;
    border-bottom: 1px solid #353330;
    padding: 24px 0;
    margin-bottom: 32px;
}

.veritas-topline {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    color: #92918d;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
}

.veritas-brand {
    margin-top: 28px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(54px, 9vw, 100px);
    font-weight: 700;
    line-height: 1;
    letter-spacing: -5px;
    color: #f2f0eb;
}

.veritas-brand span {
    color: #ff453a;
}

.veritas-subtitle {
    margin-top: 18px;
    color: #92918d;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 1.9;
    letter-spacing: 1px;
}

.veritas-footer {
    border-top: 1px solid #353330;
    margin-top: 45px;
    padding-top: 15px;
    color: #92918d;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# Store prediction history during the current session
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# 3. SIDEBAR
# -------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["News Detector", "Prediction History", "Model Information"]
)
st.markdown("""
<div class="veritas-footer">
    VERITAS / 001 · TF-IDF + LOGISTIC REGRESSION
</div>
""", unsafe_allow_html=True)

# -------------------------------
# 4. NEWS DETECTOR
# -------------------------------

if page == "News Detector":

    st.header("Analyze a News Article")

    headline = st.text_input(
        "News headline",
        placeholder="Enter the headline (optional)"
    )

    article = st.text_area(
        "News article",
        height=250,
        placeholder="Paste the full news article here..."
    )

    if st.button("Analyze News", type="primary"):

        combined_text = (headline.strip() + " " + article.strip()).strip()

        if len(combined_text.split()) < 10:
            st.warning(
                "Please enter at least 10 words of news text."
            )

        else:
            # Transform input using the trained vectorizer
            text_vector = vectorizer.transform([combined_text])

            # Predict the class
            
            # Get estimated probabilities for both classes
            probabilities = model.predict_proba(text_vector)[0]
            classes = list(model.classes_)

            fake_probability = float(
                probabilities[classes.index(0)] * 100
            )
            real_probability = float(
                probabilities[classes.index(1)] * 100
            )

            confidence = max(fake_probability, real_probability)

            # Apply an uncertainty threshold
            if confidence < 60:
                predicted_label = "UNCERTAIN"
            elif fake_probability > real_probability:
                predicted_label = "FAKE"
            else:
                predicted_label = "REAL"

            # Display results
            st.subheader("Prediction Result")

            if predicted_label == "UNCERTAIN":
                st.warning(
                    "Prediction: UNCERTAIN — the model is not confident enough."
                )
                st.write(
                    "The model's estimated probabilities are too close "
                    "to make a stronger classification."
                )
            elif predicted_label == "FAKE":
                st.error("Prediction: FAKE NEWS")
            else:
                st.success("Prediction: REAL NEWS")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fake-class probability",
                    f"{fake_probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Real-class probability",
                    f"{real_probability:.2f}%"
                )

            st.caption(
                f"Highest estimated class probability: {confidence:.2f}%. "
                "These estimates reflect learned text patterns; they do "
                "not prove whether a claim is true or false."
            )

            st.progress(int(round(confidence)))

            with st.expander("How was this result produced?"):
                st.write(
                    "1. The headline and article were combined."
                )
                st.write(
                    "2. TF-IDF converted the text into numerical features."
                )
                st.write(
                    "3. Logistic Regression estimated probabilities "
                    "using patterns learned during training."
                )
                st.write(
                    "4. If the highest estimated probability is below "
                    "60%, the app displays Uncertain."
                )

            # Add prediction to session history
            st.session_state.history.append({
                "Headline": headline.strip() or "(No headline)",
                "Prediction": predicted_label,
                "Fake probability (%)": round(fake_probability, 2),
                "Real probability (%)": round(real_probability, 2)
            })

# -------------------------------
# 5. PREDICTION HISTORY
# -------------------------------

elif page == "Prediction History":

    st.header("Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History as CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No predictions yet. Analyze an article to create history."
        )

# -------------------------------
# 6. MODEL INFORMATION
# -------------------------------


elif page == "Model Information":

    import json
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    st.header("Model Information")

    st.subheader("Dataset")

    st.write("Dataset used: Fake and Real News Dataset (ISOT).")
    st.write("Total articles loaded: 44,898.")
    st.write("Fake-labelled articles: 23,481.")
    st.write("Real-labelled articles: 21,417.")

    st.subheader("Machine-Learning Pipeline")

    st.write("1. Load and combine the labelled datasets.")
    st.write("2. Split the data into 80% training and 20% testing.")
    st.write("3. Convert text into TF-IDF numerical features.")
    st.write("4. Train a Logistic Regression classifier.")
    st.write("5. Evaluate predictions on the held-out test split.")

    st.subheader("Test Evaluation Results")

    try:
        with open("evaluation_metrics.json", "r") as file:
            metrics = json.load(file)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
            st.metric("Precision (REAL)", f"{metrics['precision']:.2f}%")

        with col2:
            st.metric("Recall (REAL)", f"{metrics['recall']:.2f}%")
            st.metric("F1-score (REAL)", f"{metrics['f1_score']:.2f}%")

        st.subheader("Confusion Matrix")

        import numpy as np

        cm = np.array(metrics["confusion_matrix"])

        fig, ax = plt.subplots()
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["FAKE", "REAL"]
        )
        display.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Fake News Detection - Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Rows represent actual labels; columns represent predicted labels."
        )

    except FileNotFoundError:
        st.error(
            "Evaluation results not found. Run evaluate_model.py first."
        )

    st.info(
        "These results measure performance on a held-out test split. "
        "They do not guarantee real-world performance."
    )

    st.subheader("Limitations")

    st.write("- The model does not search the internet for corroborating evidence.")
    st.write("- It may learn publisher-specific or dataset-specific patterns.")
    st.write("- It may misclassify satire, breaking news, and unfamiliar topics.")
    st.write("- High confidence does not guarantee a correct prediction.")

st.markdown("""
<div class="veritas-header">
    <div class="veritas-topline">
        <span>VERITAS / NEWS INTELLIGENCE</span>
        <span>RESEARCH DIVISION · 001</span>
    </div>
    <div class="veritas-brand">TRUTH<span>?</span></div>
    <div class="veritas-subtitle">
        TEXT CLASSIFICATION SYSTEM<br>
        ANALYZE THE SIGNAL. QUESTION THE SOURCE.
    </div>
</div>
""", unsafe_allow_html=True)
