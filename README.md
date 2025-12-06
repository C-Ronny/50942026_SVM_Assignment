# 📧 SMS Spam Detection System

A production-ready machine learning web application for detecting spam SMS messages with multi-model comparison and real-time prediction capabilities.


## Live Demo

**[https://mlspamdetectionapp-65ytzvpxuuk4tyjjwdyyjp.streamlit.app/](#)**

## Overview

This project implements a complete spam detection pipeline using multiple machine learning algorithms. The interactive Streamlit interface allows users to test individual messages, perform batch analysis, and compare predictions across different models in real-time.

### Key Features

- **Multi-Model Architecture**: Compare Naive Bayes, Complement NB, and Logistic Regression
- **Real-Time Prediction**: Instant spam/ham classification with confidence scores
- **Batch Testing**: Process multiple messages simultaneously
- **Model Comparison Mode**: Side-by-side prediction analysis with consensus voting
- **Interactive Visualizations**: Dynamic charts using Plotly
- **CSV Export**: Download batch results for further analysis
- **Adjustable Threshold**: Fine-tune sensitivity for precision/recall trade-offs

## Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes (Multinomial) | 98.2% | 95.6% | 93.2% | 94.4% |
| Complement NB | **98.7%** | 96.8% | 94.5% | 95.6% |
| Logistic Regression | 97.8% | 94.2% | 91.8% | 93.0% |

## Usage Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the application**
```bash
streamlit run spam_detector_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## To Use:

### Single Model Testing

1. Select **"Single Model"** mode in the sidebar
2. Choose your preferred model from the dropdown
3. Enter an SMS message in the text area
4. Click **"Analyze Message"** to see the prediction

### Multi-Model Comparison

1. Select **"Compare Models"** mode in the sidebar
2. Choose multiple models to compare (checkboxes)
3. Test messages to see predictions from all selected models
4. View consensus voting and confidence comparisons

### Batch Testing

1. Navigate to the **"Batch Testing"** tab
2. Enter multiple messages (one per line)
3. Click **"Analyze Batch"** for results
4. Download results as CSV for further analysis

## Stack Used

**Machine Learning**
- scikit-learn - Model training and evaluation
- TF-IDF Vectorization - Feature extraction
- GridSearchCV - Hyperparameter optimization

**Web Framework**
- Streamlit - Interactive web interface
- Plotly - Data visualizations

**Data Processing**
- Pandas - Data manipulation
- NumPy - Numerical operations

## 📁 Project Structure

```
sms-spam-detector/
├── spam_detector_app.py      # Main Streamlit application
├── train_models.py            # Model training pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── models/                    # Trained model files
│   ├── best_nb_pipeline.pkl
│   ├── best_cnb_pipeline.pkl
│   └── logreg_optimals_pipeline.pkl
└── data/
    └── spam_assignment.xlsx   # Training dataset
```

## Model Details

### Naive Bayes (Multinomial)
- Fast probabilistic classifier
- Assumes feature independence
- Excellent baseline performance

### Complement Naive Bayes
- Optimized for imbalanced datasets
- Grid search tuned hyperparameters
- Best overall accuracy (98.7%)

### Logistic Regression
- Linear classification model
- L2 regularization
- Interpretable feature weights


## Example Messages

**Spam Examples:**
- "FREE entry in 2 a wkly comp to win FA Cup final tickets!"
- "WINNER!! You've been selected to receive a £900 prize!"
- "URGENT! Claim your cash prize now by calling..."

**Legitimate Examples:**
- "Hey, are you free for lunch today?"
- "Meeting moved to 3pm. See you there!"
- "Thanks for your help yesterday!"