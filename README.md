# 📧 SMS Spam Detector - Streamlit App

A professional machine learning web application for detecting spam SMS messages using multiple classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🌟 Features

- **Multiple ML Models**: Naive Bayes, Complement NB, Logistic Regression
- **Real-time Prediction**: Instant spam/ham classification
- **Confidence Scores**: See probability distributions for predictions
- **Batch Testing**: Test multiple messages simultaneously
- **Interactive Visualizations**: Beautiful charts using Plotly
- **Model Comparison**: Compare performance across different algorithms
- **Example Messages**: Pre-loaded spam/ham examples for testing

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- The spam dataset file: `spam_assignment.xlsx`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Models

Before running the app, you need to train the models. Make sure `spam_assignment.xlsx` is in the same directory, then run:

```bash
python train_models.py
```

This will:
- Load and preprocess the spam dataset
- Train multiple ML models (Naive Bayes, Complement NB, Logistic Regression)
- Save trained models to the `models/` directory
- Display training accuracy for each model

Expected output:
```
🚀 SMS SPAM DETECTION - MODEL TRAINING
📊 Loading data...
   Loaded 5572 messages
   Spam: 747
   Ham: 4825
...
📊 TRAINING SUMMARY
Naive Bayes (Multinomial):  0.9820
Complement Naive Bayes:     0.9866
Logistic Regression:        0.9775
```

### 3. Run the Streamlit App

```bash
streamlit run spam_detector_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📱 Using the App

### Single Message Testing
1. Navigate to the "🔍 Single Message Test" tab
2. Enter or paste an SMS message in the text area
3. Click "🔎 Analyze Message"
4. View the prediction result with confidence score

### Batch Testing
1. Go to the "📝 Batch Testing" tab
2. Enter multiple messages (one per line)
3. Click "🔎 Analyze Batch"
4. View results in a table and download as CSV

### Model Settings
- **Model Selection**: Choose between Naive Bayes, Complement NB, or Logistic Regression
- **Threshold Adjustment**: Fine-tune the classification threshold (default: 0.5)

### Examples Tab
Test pre-loaded spam and legitimate messages to see how models perform on typical examples.

## 📂 Project Structure

```
├── spam_detector_app.py      # Main Streamlit application
├── train_models.py            # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── spam_assignment.xlsx       # Dataset (you need to provide this)
└── models/                    # Saved models directory (created after training)
    ├── naive_bayes_spam_model.pkl
    ├── best_cnb_pipeline.pkl
    ├── logistic_regression.pkl
    └── tfidf_vectorizer.pkl
```

## 🧠 Models Implemented

### 1. Multinomial Naive Bayes
- Classic probabilistic classifier
- Fast training and prediction
- Works well with TF-IDF features

### 2. Complement Naive Bayes
- Variant optimized for imbalanced datasets
- Grid search hyperparameter tuning
- Best overall performance

### 3. Logistic Regression
- Linear classification model
- Interpretable coefficients
- Robust baseline model

## 🎯 Model Performance

After training, you can expect the following approximate performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes (Multinomial) | ~98.2% | ~95.6% | ~93.2% | ~94.4% |
| Complement NB | ~98.7% | ~96.8% | ~94.5% | ~95.6% |
| Logistic Regression | ~97.8% | ~94.2% | ~91.8% | ~93.0% |

## 🛠️ Customization

### Adjusting Model Parameters

Edit `train_models.py` to modify hyperparameters:

```python
# Example: Change Logistic Regression parameters
model = LogisticRegression(
    max_iter=1000,  # Increase iterations
    C=1.0,          # Regularization strength
    random_state=42
)
```

### Adding New Features

Modify the preprocessing in `train_models.py`:

```python
# Example: Add custom text features
data['message_length'] = data['message'].apply(len)
data['word_count'] = data['message'].apply(lambda x: len(x.split()))
```

## 🐛 Troubleshooting

### Models not loading?
- Make sure you've run `train_models.py` first
- Check that the `models/` directory exists and contains `.pkl` files

### Import errors?
```bash
pip install --upgrade -r requirements.txt
```

### Dataset not found?
- Ensure `spam_assignment.xlsx` is in the same directory as the scripts
- Check the file name matches exactly (case-sensitive)

## 📊 Dataset Information

The app expects a dataset with the following structure:
- Column `v1`: Label ('ham' or 'spam')
- Column `v2`: Message text

Example:
```
v1    | v2
------|--------------------------------------------
ham   | Hey, are you free for lunch today?
spam  | FREE entry in 2 a wkly comp to win...
```

## 🤝 Contributing

Feel free to fork this project and customize it for your needs. Some ideas:
- Add more ML models (SVM, Random Forest, Neural Networks)
- Implement feature engineering pipeline
- Add message visualization features
- Create API endpoints

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Ronny** - Computer Science Student
- 🎓 CS433: Operating Systems
- 📧 Spam Detection ML Project

## 🙏 Acknowledgments

- SMS Spam Collection Dataset
- Scikit-learn library
- Streamlit framework
- Plotly visualization library

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**