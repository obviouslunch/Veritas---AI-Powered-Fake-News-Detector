# VERITAS — AI-Powered Fake News Detector

**An intelligent news classification system powered by Machine Learning.**

VERITAS is an AI-powered web application designed to classify news articles as **REAL** or **FAKE** using Natural Language Processing (NLP) and Machine Learning. Built with Python and Streamlit, it provides an interactive interface for analyzing news content and displaying prediction confidence scores.

## Features

- **Fake News Detection:** Classifies news articles as real or fake using machine learning.
- **Confidence Scores:** Displays estimated probabilities for each classification.
- **Uncertainty Detection:** Highlights predictions where the model has low confidence.
- **Prediction History:** Allows users to review previous predictions within the current session.
- **Model Information:** Displays model performance metrics and a confusion matrix.
- **Interactive Interface:** A forensic-inspired dark interface with red accents.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application framework |
| Scikit-learn | Machine learning model development |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| TF-IDF | Text feature extraction |
| Logistic Regression | News classification |
| Joblib | Model serialization |

## Dataset

The project uses the **ISOT Fake and Real News Dataset**, containing labeled news articles.

- Original dataset: 44,898 articles
- Data preprocessing: Cleaning, removing duplicates, and preparing text
- Training and testing split: 80% training, 20% testing

The dataset provides examples of real and fake news articles used to train and evaluate the classification model.

## Machine Learning Methodology

**1. Data Preprocessing**

News articles are cleaned and prepared for machine learning.

**2. Feature Extraction**

TF-IDF (Term Frequency–Inverse Document Frequency) converts textual information into numerical feature vectors.

**3. Model Training**

A Logistic Regression classifier is trained on labeled news articles to recognize patterns associated with real and fake news.

**4. Prediction**

User-provided headlines and article content are transformed using the trained TF-IDF vectorizer and classified by the model.

**5. Confidence Estimation**

The application displays estimated class probabilities and highlights uncertain predictions.

## Model Performance

The trained classifier achieved approximately **98% accuracy** on a held-out test dataset.

Model evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

*Note: High accuracy on the test dataset does not guarantee reliable predictions on all real-world news articles.*

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/VERITAS-Fake-News-Detector.git
cd VERITAS-Fake-News-Detector
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the virtual environment**

Windows:

```bash
venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
streamlit run app.py
```

The application will open in your browser.

## Application Interface

VERITAS includes three main sections:

**News Detector:** Enter a headline and news article to receive a classification and confidence score.

**Prediction History:** View predictions made during the current session.

**Model Information:** Explore the machine learning methodology, performance metrics, and confusion matrix.

## Live Demo

**Application:** Coming soon

## Limitations

VERITAS identifies textual patterns learned from its training dataset. It does not independently verify factual claims, consult live news sources, or guarantee that its classifications are correct.

## Future Improvements

- Integration with real-time news verification APIs
- Support for multiple languages
- Improved generalization to recent news articles
- Comparison with additional machine learning models
- Enhanced explainability of predictions

## Project Information

**Project Name:** VERITAS — AI-Powered Fake News Detector

**Project Type:** Machine Learning / Natural Language Processing / Web Application

**Developed Using:** Python, Scikit-learn, and Streamlit

**Purpose:** Academic project demonstrating the application of machine learning techniques to automated news classification.
