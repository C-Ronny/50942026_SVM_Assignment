import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .spam-message {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .ham-message {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

class SpamDetector:
    """Main spam detection class"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.vectorizers = {}
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        if not self.models_dir.exists():
            st.warning("⚠️ Models directory not found. Please train models first.")
            return
        
        # Try to load different model types
        model_files = {
            'Naive Bayes': 'naive_bayes_spam_model.pkl',
            'Complement NB': 'best_cnb_pipeline.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
        }
        
        for name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    self.models[name] = joblib.load(model_path)
                except Exception as e:
                    st.warning(f"Could not load {name}: {e}")
        
        # Load vectorizers
        vectorizer_path = self.models_dir / 'tfidf_vectorizer.pkl'
        if vectorizer_path.exists():
            try:
                self.vectorizers['default'] = joblib.load(vectorizer_path)
            except Exception as e:
                st.warning(f"Could not load vectorizer: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text message"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters (keep letters and spaces)
        text = re.sub(r"[^a-z'\s]", '', text)
        return text
    
    def predict(self, text, model_name, threshold=0.5):
        """Make prediction on text"""
        if model_name not in self.models:
            return None, None
        
        model = self.models[model_name]
        processed_text = self.preprocess_text(text)
        
        # Check if model is a pipeline (has predict method directly)
        try:
            # For pipeline models (like best_cnb_pipeline)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([processed_text])[0]
                prediction = 1 if proba[1] >= threshold else 0
            else:
                # For models that need vectorization
                if 'default' not in self.vectorizers:
                    return None, None
                
                vectorizer = self.vectorizers['default']
                text_vector = vectorizer.transform([processed_text])
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(text_vector)[0]
                else:
                    # For models without predict_proba
                    prediction = model.predict(text_vector)[0]
                    proba = [1-prediction, prediction]
                
                prediction = 1 if proba[1] >= threshold else 0
            
            return prediction, proba
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

def create_probability_chart(proba, prediction):
    """Create a probability visualization"""
    labels = ['Ham (Legitimate)', 'Spam']
    values = [proba[0] * 100, proba[1] * 100]
    colors = ['#4caf50', '#f44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability (%)",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_feature_importance_chart():
    """Create a sample feature importance chart"""
    features = ['Text Length', 'Spam Words', 'Word Diversity', 'Short Words', 
                'Long Words', 'Repeated Words', 'Action Words', 'Has Greeting']
    importance = [0.25, 0.35, 0.15, 0.08, 0.05, 0.06, 0.04, 0.02]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color='#1f77b4',
            text=[f'{v:.2f}' for v in importance],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Feature Importance (Sample)",
        xaxis_title="Importance Score",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📧 SMS Spam Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize detector
    detector = SpamDetector()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        available_models = list(detector.models.keys())
        if not available_models:
            st.error("No models loaded! Please train models first.")
            st.info("Run the training script to generate models.")
            return
        
        # Mode selection: Single or Compare
        test_mode = st.radio(
            "Testing Mode",
            ["Single Model", "Compare Models"],
            help="Choose to test with one model or compare multiple models"
        )
        
        if test_mode == "Single Model":
            selected_models = [st.selectbox(
                "Select Model",
                available_models,
                help="Choose which model to use for prediction"
            )]
        else:
            selected_models = st.multiselect(
                "Select Models to Compare",
                available_models,
                default=available_models,
                help="Choose multiple models to compare predictions"
            )
            if not selected_models:
                st.warning("Please select at least one model")
                selected_models = [available_models[0]]
        
        # Threshold adjustment
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability threshold for spam classification"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Info")
        if test_mode == "Single Model":
            st.info(f"**Active Model:** {selected_models[0]}")
        else:
            st.info(f"**Comparing:** {len(selected_models)} models")
            for model in selected_models:
                st.write(f"• {model}")
        st.info(f"**Total Available:** {len(available_models)}")
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.write("""
            This app uses machine learning to classify SMS messages as spam or legitimate (ham).
            
            **Features:**
            - Multiple ML models (Naive Bayes, Logistic Regression, SVM)
            - Real-time prediction
            - Confidence scores
            - Batch testing capability
            
            **Models trained on:**
            - 5,000+ SMS messages
            - TF-IDF vectorization
            - Multiple engineered features
            """)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Message Test", 
        "📝 Batch Testing", 
        "📊 Model Performance",
        "💡 Examples"
    ])
    
    # Tab 1: Single message testing
    with tab1:
        st.header("Test a Single Message")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            message = st.text_area(
                "Enter your message:",
                height=150,
                placeholder="Type or paste an SMS message here...",
                help="Enter the text message you want to classify"
            )
            
            # Predict button
            if st.button("🔎 Analyze Message", type="primary", use_container_width=True):
                if message.strip():
                    with st.spinner("Analyzing..."):
                        st.markdown("### Results")
                        
                        if test_mode == "Compare Models":
                            # Compare multiple models
                            st.markdown("#### 📊 Model Comparison")
                            
                            results = []
                            for model_name in selected_models:
                                prediction, proba = detector.predict(message, model_name, threshold)
                                if prediction is not None:
                                    results.append({
                                        'Model': model_name,
                                        'Prediction': 'SPAM' if prediction == 1 else 'HAM',
                                        'Spam Confidence': f"{proba[1]*100:.1f}%",
                                        'Ham Confidence': f"{proba[0]*100:.1f}%",
                                        'spam_prob': proba[1]
                                    })
                            
                            if results:
                                # Create comparison table
                                df_results = pd.DataFrame(results)
                                st.dataframe(df_results[['Model', 'Prediction', 'Spam Confidence', 'Ham Confidence']], 
                                           use_container_width=True, hide_index=True)
                                
                                # Visual comparison
                                st.markdown("#### 📈 Visual Comparison")
                                
                                # Create comparison chart
                                fig = go.Figure()
                                for i, row in enumerate(results):
                                    model_name = row['Model']
                                    spam_prob = row['spam_prob']
                                    ham_prob = 1 - spam_prob
                                    
                                    fig.add_trace(go.Bar(
                                        name=model_name,
                                        x=['Ham', 'Spam'],
                                        y=[ham_prob * 100, spam_prob * 100],
                                        text=[f'{ham_prob*100:.1f}%', f'{spam_prob*100:.1f}%'],
                                        textposition='auto',
                                    ))
                                
                                fig.update_layout(
                                    title="Prediction Confidence Comparison",
                                    yaxis_title="Probability (%)",
                                    barmode='group',
                                    height=400,
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Consensus analysis
                                spam_count = sum(1 for r in results if r['Prediction'] == 'SPAM')
                                ham_count = len(results) - spam_count
                                
                                st.markdown("#### 🎯 Consensus")
                                if spam_count > ham_count:
                                    st.error(f"**Majority Vote: SPAM** ({spam_count}/{len(results)} models)")
                                elif ham_count > spam_count:
                                    st.success(f"**Majority Vote: HAM** ({ham_count}/{len(results)} models)")
                                else:
                                    st.warning(f"**Split Decision** ({spam_count} SPAM, {ham_count} HAM)")
                        
                        else:
                            # Single model prediction
                            prediction, proba = detector.predict(message, selected_models[0], threshold)
                            
                            if prediction is not None:
                                # Result display
                                if prediction == 1:
                                    st.markdown(
                                        f'<div class="spam-message"><h3>🚨 SPAM DETECTED</h3>'
                                        f'<p>Confidence: {proba[1]*100:.1f}%</p></div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="ham-message"><h3>✅ LEGITIMATE MESSAGE</h3>'
                                        f'<p>Confidence: {proba[0]*100:.1f}%</p></div>',
                                        unsafe_allow_html=True
                                    )
                                
                                # Probability chart
                                st.plotly_chart(
                                    create_probability_chart(proba, prediction),
                                    use_container_width=True
                                )
                        
                        # Message statistics (for both modes)
                        with st.expander("📈 Message Statistics"):
                            col_a, col_b, col_c, col_d = st.columns(4)
                            col_a.metric("Characters", len(message))
                            col_b.metric("Words", len(message.split()))
                            col_c.metric("Avg Word Length", f"{len(message)/max(len(message.split()), 1):.1f}")
                            col_d.metric("Unique Words", len(set(message.lower().split())))
                else:
                    st.warning("⚠️ Please enter a message to analyze.")
        
        with col2:
            st.subheader("💡 Quick Stats")
            
            # Display some helpful metrics
            if test_mode == "Single Model":
                st.metric("Active Model", selected_models[0])
            else:
                st.metric("Models Testing", len(selected_models))
            st.metric("Threshold", f"{threshold:.2f}")
            
            if message:
                spam_words = ['free', 'win', 'winner', 'cash', 'prize', 'claim', 
                             'call', 'text', 'urgent', 'now']
                spam_count = sum(1 for word in spam_words if word in message.lower())
                st.metric("Spam Keywords", spam_count)
    
    # Tab 2: Batch testing
    with tab2:
        st.header("Batch Message Testing")
        
        if test_mode == "Compare Models":
            st.info(f"📊 Testing with {len(selected_models)} models: {', '.join(selected_models)}")
        else:
            st.info(f"📊 Testing with: {selected_models[0]}")
        
        st.write("Test multiple messages at once by entering them below (one per line)")
        
        batch_input = st.text_area(
            "Enter messages (one per line):",
            height=200,
            placeholder="Free entry in 2 a wkly comp...\nHey, are we still on for dinner?\nCongratulations! You've won £1000..."
        )
        
        if st.button("🔎 Analyze Batch", type="primary"):
            if batch_input.strip():
                messages = [msg.strip() for msg in batch_input.split('\n') if msg.strip()]
                
                if test_mode == "Compare Models":
                    # Compare models on batch
                    st.subheader("📊 Model Comparison Results")
                    
                    all_results = []
                    for msg in messages:
                        msg_results = {'Message': msg[:50] + '...' if len(msg) > 50 else msg}
                        
                        for model_name in selected_models:
                            prediction, proba = detector.predict(msg, model_name, threshold)
                            if prediction is not None:
                                msg_results[f'{model_name} Prediction'] = 'SPAM' if prediction == 1 else 'HAM'
                                msg_results[f'{model_name} Confidence'] = f"{max(proba)*100:.1f}%"
                        
                        all_results.append(msg_results)
                    
                    if all_results:
                        df = pd.DataFrame(all_results)
                        
                        # Summary metrics per model
                        st.markdown("#### 📈 Summary by Model")
                        summary_cols = st.columns(len(selected_models))
                        
                        for i, model_name in enumerate(selected_models):
                            with summary_cols[i]:
                                pred_col = f'{model_name} Prediction'
                                if pred_col in df.columns:
                                    spam_count = (df[pred_col] == 'SPAM').sum()
                                    ham_count = (df[pred_col] == 'HAM').sum()
                                    st.metric(model_name, f"{spam_count} spam")
                                    st.caption(f"{ham_count} ham")
                        
                        # Full results table
                        st.markdown("#### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "spam_detection_comparison.csv",
                            "text/csv",
                            key='download-csv-compare'
                        )
                
                else:
                    # Single model batch testing
                    results = []
                    for msg in messages:
                        prediction, proba = detector.predict(msg, selected_models[0], threshold)
                        if prediction is not None:
                            results.append({
                                'Message': msg[:50] + '...' if len(msg) > 50 else msg,
                                'Classification': 'SPAM' if prediction == 1 else 'HAM',
                                'Confidence': f"{max(proba)*100:.1f}%",
                                'Spam Probability': proba[1]
                            })
                    
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Summary metrics
                        st.subheader("📊 Batch Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Messages", len(results))
                        col2.metric("Spam Detected", sum(1 for r in results if r['Classification'] == 'SPAM'))
                        col3.metric("Legitimate", sum(1 for r in results if r['Classification'] == 'HAM'))
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "spam_detection_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
            else:
                st.warning("⚠️ Please enter at least one message.")
    
    # Tab 3: Model performance
    with tab3:
        st.header("Model Performance Metrics")
        
        st.info("📌 These are sample metrics. Train models with the training script to see actual performance.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Performance Metrics")
            
            # Sample metrics for demonstration
            metrics_data = {
                'Model': available_models[:3] if len(available_models) >= 3 else available_models,
                'Accuracy': [0.982, 0.975, 0.968][:len(available_models[:3])],
                'Precision': [0.956, 0.948, 0.942][:len(available_models[:3])],
                'Recall': [0.932, 0.925, 0.918][:len(available_models[:3])],
                'F1-Score': [0.944, 0.936, 0.930][:len(available_models[:3])]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Bar chart comparison
            fig = px.bar(
                metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Metric',
                y='Score',
                color='Model',
                barmode='group',
                title="Model Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Feature Analysis")
            st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
    
    # Tab 4: Examples
    with tab4:
        st.header("Example Messages")
        
        if test_mode == "Compare Models":
            st.info(f"🔬 Comparing with {len(selected_models)} models")
        
        st.write("Try these example messages to see how the model performs:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Spam Examples")
            spam_examples = [
                "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
                "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
                "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw",
                "URGENT! You have won a 1week FREE membership in our £100000 Prize Jackpot!",
                "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera"
            ]
            
            for i, example in enumerate(spam_examples, 1):
                if st.button(f"Test Spam #{i}", key=f"spam_{i}"):
                    st.write(f"**Message:** {example}")
                    
                    if test_mode == "Compare Models":
                        st.markdown("**Predictions:**")
                        for model_name in selected_models:
                            prediction, proba = detector.predict(example, model_name, threshold)
                            if prediction is not None:
                                result = "🚨 SPAM" if prediction == 1 else "✅ HAM"
                                st.write(f"• **{model_name}:** {result} ({proba[1]*100:.1f}% spam)")
                    else:
                        prediction, proba = detector.predict(example, selected_models[0], threshold)
                        if prediction is not None:
                            result = "SPAM" if prediction == 1 else "HAM"
                            st.write(f"**Prediction:** {result} ({proba[1]*100:.1f}% spam probability)")
        
        with col2:
            st.subheader("✅ Legitimate Examples")
            ham_examples = [
                "Hey, are you free for lunch today?",
                "Meeting moved to 3pm. See you in conference room B.",
                "Thanks for your help yesterday. Really appreciate it!",
                "Don't forget to pick up milk on your way home",
                "Happy birthday! Hope you have a wonderful day"
            ]
            
            for i, example in enumerate(ham_examples, 1):
                if st.button(f"Test Ham #{i}", key=f"ham_{i}"):
                    st.write(f"**Message:** {example}")
                    
                    if test_mode == "Compare Models":
                        st.markdown("**Predictions:**")
                        for model_name in selected_models:
                            prediction, proba = detector.predict(example, model_name, threshold)
                            if prediction is not None:
                                result = "🚨 SPAM" if prediction == 1 else "✅ HAM"
                                st.write(f"• **{model_name}:** {result} ({proba[0]*100:.1f}% ham)")
                    else:
                        prediction, proba = detector.predict(example, selected_models[0], threshold)
                        if prediction is not None:
                            result = "SPAM" if prediction == 1 else "HAM"
                            st.write(f"**Prediction:** {result} ({proba[0]*100:.1f}% ham probability)")

    # Footer
    st.markdown("---")
    footer_text = f"Built with Streamlit | Machine Learning Spam Detector | "
    if test_mode == "Compare Models":
        footer_text += f"Comparing {len(selected_models)} models"
    else:
        footer_text += f"Using {selected_models[0]}"
    
    st.markdown(
        f"<div style='text-align: center; color: #666;'>{footer_text}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()