"""
Streamlit dashboard for fraud detection system.

This module provides a comprehensive web-based dashboard for fraud analysis,
including data upload and exploration, model training and evaluation,
and real-time fraud detection capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Import fraud detection components
from src.data.data_loader import DataLoader
from src.data.data_explorer import DataExplorer
from src.data.data_cleaner import DataCleaner
from src.data.feature_engineering import FeatureEngineering
from src.data.data_encoder import DataEncoder
from src.models.logistic_regression_model import LogisticRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.model_evaluator import ModelEvaluator
from src.visualization.data_visualizer import DataVisualizer
from src.visualization.model_visualizer import ModelVisualizer
from src.visualization.fraud_pattern_analyzer import FraudPatternAnalyzer
from src.services.fraud_detector import FraudDetector
from src.services.alert_manager import AlertManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FraudDashboard:
    """Main dashboard class for fraud detection system."""
    
    def __init__(self):
        """Initialize the dashboard with necessary components."""
        self.data_loader = DataLoader()
        self.data_explorer = DataExplorer()
        self.data_cleaner = DataCleaner()
        self.feature_engineering = FeatureEngineering()
        self.data_encoder = DataEncoder()
        self.model_evaluator = ModelEvaluator()
        self.data_visualizer = DataVisualizer()
        self.model_visualizer = ModelVisualizer()
        self.fraud_pattern_analyzer = FraudPatternAnalyzer()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'model_results' not in st.session_state:
            st.session_state.model_results = {}
        if 'fraud_detector' not in st.session_state:
            st.session_state.fraud_detector = None
        if 'alert_manager' not in st.session_state:
            st.session_state.alert_manager = None
    
    def run(self):
        """Run the main dashboard application."""
        # Main header
        st.markdown('<h1 class="main-header">🔍 Fraud Detection Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Data Upload & Exploration", "Model Training & Evaluation", "Real-time Fraud Detection"]
        )
        
        # Route to appropriate page
        if page == "Data Upload & Exploration":
            self.data_exploration_page()
        elif page == "Model Training & Evaluation":
            self.model_training_page()
        elif page == "Real-time Fraud Detection":
            self.fraud_detection_page()
    
    def data_exploration_page(self):
        """Data upload and exploration interface."""
        st.markdown('<h2 class="section-header">📊 Data Upload & Exploration</h2>', 
                   unsafe_allow_html=True)
        
        # Data upload section
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your fraud detection dataset in CSV format"
        )
        
        # Sample data option
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Sample Data"):
                self._load_sample_data()
        
        with col2:
            if st.button("Clear Data"):
                self._clear_data()
        
        # Process uploaded file
        if uploaded_file is not None:
            self._process_uploaded_file(uploaded_file)
        
        # Display data exploration if data is loaded
        if st.session_state.data is not None:
            self._display_data_exploration()
    
    def model_training_page(self):
        """Model training and evaluation dashboard."""
        st.markdown('<h2 class="section-header">🤖 Model Training & Evaluation</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("Please upload and explore data first!")
            return
        
        # Data preprocessing section
        st.subheader("Data Preprocessing")
        if st.button("Preprocess Data"):
            self._preprocess_data()
        
        if st.session_state.processed_data is not None:
            st.success("Data preprocessing completed!")
            
            # Model training section
            st.subheader("Model Training")
            self._display_model_training()
            
            # Model evaluation section
            if st.session_state.trained_models:
                st.subheader("Model Evaluation")
                self._display_model_evaluation()
    
    def fraud_detection_page(self):
        """Real-time fraud detection interface."""
        st.markdown('<h2 class="section-header">🚨 Real-time Fraud Detection</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.trained_models:
            st.warning("Please train models first!")
            return
        
        # Initialize fraud detector
        if st.session_state.fraud_detector is None:
            self._initialize_fraud_detector()
        
        # Single transaction scoring
        st.subheader("Single Transaction Analysis")
        self._display_single_transaction_interface()
        
        # Batch prediction
        st.subheader("Batch Prediction")
        self._display_batch_prediction_interface()
        
        # Alert management
        st.subheader("Alert Management")
        self._display_alert_management()
    
    def _load_sample_data(self):
        """Load sample fraud detection data."""
        try:
            # Try to load from data directory
            sample_paths = [
                "data/raw/fraud_data.csv",
                "data/processed/fraud_data.csv",
                "examples/sample_fraud_data.csv"
            ]
            
            for path in sample_paths:
                try:
                    st.session_state.data = self.data_loader.load_data(path)
                    st.success(f"Sample data loaded successfully from {path}!")
                    break
                except FileNotFoundError:
                    continue
            else:
                # Generate synthetic sample data if no file found
                st.session_state.data = self._generate_sample_data()
                st.info("Generated synthetic sample data for demonstration.")
                
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate synthetic sample data for demonstration."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic transaction data
        data = {
            'step': np.random.randint(1, 745, n_samples),
            'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_samples),
            'amount': np.random.lognormal(8, 2, n_samples),
            'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
            'oldbalanceOrg': np.random.lognormal(10, 1.5, n_samples),
            'newbalanceOrig': np.random.lognormal(10, 1.5, n_samples),
            'nameDest': [f'C{i:09d}' if np.random.random() > 0.3 else f'M{i:09d}' 
                        for i in range(n_samples)],
            'oldbalanceDest': np.random.lognormal(9, 1.8, n_samples),
            'newbalanceDest': np.random.lognormal(9, 1.8, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.998, 0.002]),
            'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
        }
        
        df = pd.DataFrame(data)
        
        # Make fraud cases more realistic
        fraud_mask = df['isFraud'] == 1
        df.loc[fraud_mask, 'type'] = np.random.choice(['TRANSFER', 'CASH_OUT'], fraud_mask.sum())
        df.loc[fraud_mask, 'amount'] = np.random.lognormal(12, 1, fraud_mask.sum())
        
        return df
    
    def _clear_data(self):
        """Clear all data and reset session state."""
        st.session_state.data = None
        st.session_state.processed_data = None
        st.session_state.trained_models = {}
        st.session_state.model_results = {}
        st.session_state.fraud_detector = None
        st.success("Data cleared successfully!")
    
    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded CSV file."""
        try:
            # Read the uploaded file
            bytes_data = uploaded_file.read()
            st.session_state.data = pd.read_csv(io.StringIO(bytes_data.decode('utf-8')))
            st.success(f"File uploaded successfully! Loaded {len(st.session_state.data)} rows.")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    def _display_data_exploration(self):
        """Display data exploration interface."""
        data = st.session_state.data
        
        # Basic data information
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(data):,}")
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            if 'isFraud' in data.columns:
                fraud_count = data['isFraud'].sum()
                st.metric("Fraud Cases", f"{fraud_count:,}")
            else:
                st.metric("Fraud Cases", "N/A")
        with col4:
            if 'isFraud' in data.columns:
                fraud_rate = (data['isFraud'].sum() / len(data)) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
            else:
                st.metric("Fraud Rate", "N/A")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        
        # Data quality check
        st.subheader("Data Quality")
        missing_data = data.isnull().sum()
        if missing_data.any():
            st.warning("Missing values detected:")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(data)) * 100
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        else:
            st.success("No missing values detected!")
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # Transaction type distribution
        if 'type' in data.columns:
            type_counts = data['type'].value_counts().reset_index()
            type_counts.columns = ['Transaction Type', 'Count']
            fig_type = px.bar(
                type_counts,
                x='Transaction Type', y='Count',
                title="Transaction Type Distribution",
                labels={'Transaction Type': 'Transaction Type', 'Count': 'Count'}
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        # Amount distribution
        if 'amount' in data.columns:
            fig_amount = px.histogram(
                data, x='amount',
                title="Transaction Amount Distribution",
                nbins=50
            )
            fig_amount.update_xaxes(type="log", title="Amount (log scale)")
            st.plotly_chart(fig_amount, use_container_width=True)
        
        # Fraud patterns
        if 'isFraud' in data.columns and 'type' in data.columns:
            fraud_by_type = data.groupby(['type', 'isFraud']).size().reset_index(name='count')
            fraud_by_type['fraud_status'] = fraud_by_type['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            
            fig_fraud = px.bar(
                fraud_by_type, x='type', y='count', color='fraud_status',
                title="Fraud Distribution by Transaction Type",
                color_discrete_map={'Legitimate': 'lightblue', 'Fraud': 'red'}
            )
            st.plotly_chart(fig_fraud, use_container_width=True)
    
    def _preprocess_data(self):
        """Preprocess the data for model training."""
        try:
            with st.spinner("Preprocessing data..."):
                data = st.session_state.data.copy()
                
                # Clean data
                cleaned_data = self.data_cleaner.clean_data(data)
                
                # Feature engineering
                engineered_data = self.feature_engineering.engineer_features(cleaned_data)
                
                # Encode categorical variables
                categorical_features = ['type']  # Main categorical feature
                if 'nameOrig' in engineered_data.columns:
                    # For customer names, we'll use a simplified encoding approach
                    # Convert to merchant flags instead of encoding all unique names
                    pass  # Already handled in feature engineering
                
                encoded_data = self.data_encoder.encode_categorical_features(
                    engineered_data, categorical_features
                )
                
                # Store processed data
                st.session_state.processed_data = encoded_data
                
                st.success("Data preprocessing completed successfully!")
                
                # Show preprocessing summary
                st.subheader("Preprocessing Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Features", len(data.columns))
                    st.metric("Original Samples", len(data))
                with col2:
                    st.metric("Processed Features", len(encoded_data.columns))
                    st.metric("Processed Samples", len(encoded_data))
                
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
    
    def _display_model_training(self):
        """Display model training interface."""
        # Model selection
        st.write("Select models to train:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_lr = st.checkbox("Logistic Regression", value=True)
        with col2:
            train_rf = st.checkbox("Random Forest", value=True)
        with col3:
            train_xgb = st.checkbox("XGBoost", value=True)
        
        # Training parameters
        st.subheader("Training Parameters")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Train models button
        if st.button("Train Selected Models"):
            self._train_models(train_lr, train_rf, train_xgb, test_size, random_state)
    
    def _train_models(self, train_lr: bool, train_rf: bool, train_xgb: bool, 
                     test_size: float, random_state: int):
        """Train selected models."""
        try:
            data = st.session_state.processed_data
            
            # Prepare features and target
            if 'isFraud' not in data.columns:
                st.error("Target variable 'isFraud' not found in data!")
                return
            
            # Remove non-numerical columns and target variable
            exclude_cols = ['isFraud', 'nameOrig', 'nameDest']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            # Select only numerical features for training
            X = data[feature_cols].select_dtypes(include=[np.number])
            y = data['isFraud']
            
            st.info(f"Using {len(X.columns)} numerical features for training: {list(X.columns)[:5]}...")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Store test data for evaluation
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models_to_train = []
            if train_lr:
                models_to_train.append(('Logistic Regression', LogisticRegressionModel()))
            if train_rf:
                models_to_train.append(('Random Forest', RandomForestModel()))
            if train_xgb:
                models_to_train.append(('XGBoost', XGBoostModel()))
            
            # Train models
            for i, (name, model) in enumerate(models_to_train):
                status_text.text(f"Training {name}...")
                
                # Train model
                model.train(X_train, y_train)
                
                # Evaluate model
                results = self.model_evaluator.evaluate_model(model, X_test, y_test, name)
                
                # Store results
                st.session_state.trained_models[name] = model
                st.session_state.model_results[name] = results
                
                progress_bar.progress((i + 1) / len(models_to_train))
            
            status_text.text("Training completed!")
            st.success(f"Successfully trained {len(models_to_train)} models!")
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
    
    def _display_model_evaluation(self):
        """Display model evaluation results."""
        results = st.session_state.model_results
        
        # Performance comparison table
        st.subheader("Model Performance Comparison")
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result.get('accuracy', 0):.4f}",
                'Precision': f"{result.get('precision', 0):.4f}",
                'Recall': f"{result.get('recall', 0):.4f}",
                'F1-Score': f"{result.get('f1_score', 0):.4f}",
                'AUC-ROC': f"{result.get('auc_roc', 0):.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Performance visualization
        st.subheader("Performance Visualization")
        
        # Metrics comparison chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        model_names = list(results.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[name].get(metric, 0) for name in model_names]
            fig.add_trace(go.Scatter(
                x=model_names,
                y=values,
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        cols = st.columns(len(results))
        for i, (name, result) in enumerate(results.items()):
            with cols[i]:
                if 'confusion_matrix' in result:
                    cm = np.array(result['confusion_matrix'])
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title=f"{name}",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Legitimate', 'Fraud'],
                        y=['Legitimate', 'Fraud']
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curves
        st.subheader("ROC Curves")
        
        fig_roc = go.Figure()
        
        for name, result in results.items():
            if 'roc_curve' in result:
                roc_data = result['roc_curve']
                fig_roc.add_trace(go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    mode='lines',
                    name=f"{name} (AUC = {result.get('auc_roc', 0):.3f})",
                    line=dict(width=3)
                ))
        
        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    def _initialize_fraud_detector(self):
        """Initialize fraud detector with best model."""
        if not st.session_state.trained_models:
            return
        
        # Select best model based on F1-score
        best_model_name = None
        best_f1_score = 0
        
        for name, result in st.session_state.model_results.items():
            f1_score = result.get('f1_score', 0)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_name = name
        
        if best_model_name:
            best_model = st.session_state.trained_models[best_model_name]
            
            # Initialize alert manager
            alert_manager = AlertManager()
            st.session_state.alert_manager = alert_manager
            
            # Initialize fraud detector
            fraud_detector = FraudDetector(
                model=best_model,
                alert_manager=alert_manager
            )
            st.session_state.fraud_detector = fraud_detector
            
            st.success(f"Fraud detector initialized with {best_model_name} model!")
    
    def _display_single_transaction_interface(self):
        """Display single transaction analysis interface."""
        st.write("Enter transaction details for fraud analysis:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Time Step", min_value=1, max_value=744, value=1)
            transaction_type = st.selectbox("Transaction Type", 
                                          ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
            old_balance_org = st.number_input("Origin Old Balance", min_value=0.0, value=5000.0, format="%.2f")
            new_balance_org = st.number_input("Origin New Balance", min_value=0.0, value=4000.0, format="%.2f")
        
        with col2:
            name_orig = st.text_input("Origin Account", value="C123456789")
            name_dest = st.text_input("Destination Account", value="C987654321")
            old_balance_dest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0, format="%.2f")
            new_balance_dest = st.number_input("Destination New Balance", min_value=0.0, value=1000.0, format="%.2f")
        
        if st.button("Analyze Transaction"):
            self._analyze_single_transaction({
                'step': step,
                'type': transaction_type,
                'amount': amount,
                'nameOrig': name_orig,
                'oldbalanceOrg': old_balance_org,
                'newbalanceOrig': new_balance_org,
                'nameDest': name_dest,
                'oldbalanceDest': old_balance_dest,
                'newbalanceDest': new_balance_dest
            })
    
    def _analyze_single_transaction(self, transaction: Dict[str, Any]):
        """Analyze a single transaction for fraud."""
        try:
            fraud_detector = st.session_state.fraud_detector
            
            # Get fraud score
            fraud_score = fraud_detector.score_transaction(transaction)
            
            # Get detailed explanation
            explanation = fraud_detector.get_fraud_explanation(transaction)
            
            # Display results
            st.subheader("Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Score", f"{fraud_score:.4f}")
            with col2:
                risk_level = explanation.get('risk_level', 'UNKNOWN')
                color = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}.get(risk_level, 'gray')
                st.markdown(f"**Risk Level:** <span style='color: {color}'>{risk_level}</span>", 
                           unsafe_allow_html=True)
            with col3:
                is_fraud = explanation.get('is_fraud_prediction', False)
                prediction_text = "FRAUD" if is_fraud else "LEGITIMATE"
                prediction_color = "red" if is_fraud else "green"
                st.markdown(f"**Prediction:** <span style='color: {prediction_color}'>{prediction_text}</span>", 
                           unsafe_allow_html=True)
            
            # Risk factors
            st.subheader("Risk Factors")
            risk_factors = explanation.get('risk_factors', {})
            
            if risk_factors.get('high_risk_factors'):
                st.error("**High Risk Factors:**")
                for factor in risk_factors['high_risk_factors']:
                    st.write(f"• {factor}")
            
            if risk_factors.get('medium_risk_factors'):
                st.warning("**Medium Risk Factors:**")
                for factor in risk_factors['medium_risk_factors']:
                    st.write(f"• {factor}")
            
            if risk_factors.get('low_risk_factors'):
                st.info("**Low Risk Factors:**")
                for factor in risk_factors['low_risk_factors'][:3]:  # Limit display
                    st.write(f"• {factor}")
            
            # Explanation text
            st.subheader("Detailed Explanation")
            st.write(explanation.get('explanation_text', 'No explanation available'))
            
            # Recommendations
            recommendations = explanation.get('recommendations', [])
            if recommendations:
                st.subheader("Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")
            
        except Exception as e:
            st.error(f"Error analyzing transaction: {str(e)}")
    
    def _display_batch_prediction_interface(self):
        """Display batch prediction interface."""
        st.write("Upload a CSV file with transactions for batch analysis:")
        
        batch_file = st.file_uploader("Choose CSV file for batch prediction", type="csv")
        
        if batch_file is not None:
            try:
                # Read batch data
                batch_data = pd.read_csv(batch_file)
                st.write(f"Loaded {len(batch_data)} transactions for analysis")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(batch_data.head())
                
                if st.button("Run Batch Analysis"):
                    self._run_batch_prediction(batch_data)
                    
            except Exception as e:
                st.error(f"Error loading batch file: {str(e)}")
    
    def _run_batch_prediction(self, batch_data: pd.DataFrame):
        """Run batch prediction on uploaded data."""
        try:
            fraud_detector = st.session_state.fraud_detector
            
            with st.spinner("Running batch analysis..."):
                # Run batch prediction
                results = fraud_detector.batch_predict(batch_data)
                
                st.success(f"Batch analysis completed for {len(results)} transactions!")
                
                # Display summary
                st.subheader("Batch Analysis Summary")
                
                fraud_count = (results['fraud_prediction'] == 1).sum()
                high_risk_count = (results['risk_level'] == 'HIGH').sum()
                medium_risk_count = (results['risk_level'] == 'MEDIUM').sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(results))
                with col2:
                    st.metric("Predicted Fraud", fraud_count)
                with col3:
                    st.metric("High Risk", high_risk_count)
                with col4:
                    st.metric("Medium Risk", medium_risk_count)
                
                # Display results table
                st.subheader("Detailed Results")
                
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    risk_filter = st.selectbox("Filter by Risk Level", 
                                             ['All', 'HIGH', 'MEDIUM', 'LOW'])
                with col2:
                    fraud_filter = st.selectbox("Filter by Prediction", 
                                              ['All', 'Fraud', 'Legitimate'])
                
                # Apply filters
                filtered_results = results.copy()
                if risk_filter != 'All':
                    filtered_results = filtered_results[filtered_results['risk_level'] == risk_filter]
                if fraud_filter == 'Fraud':
                    filtered_results = filtered_results[filtered_results['fraud_prediction'] == 1]
                elif fraud_filter == 'Legitimate':
                    filtered_results = filtered_results[filtered_results['fraud_prediction'] == 0]
                
                # Display filtered results
                st.dataframe(filtered_results)
                
                # Download results
                csv = filtered_results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"fraud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error running batch prediction: {str(e)}")
    
    def _display_alert_management(self):
        """Display alert management interface."""
        alert_manager = st.session_state.alert_manager
        
        if alert_manager is None:
            st.warning("Alert manager not initialized!")
            return
        
        # Alert statistics
        st.subheader("Alert Statistics")
        
        try:
            alert_stats = alert_manager.get_alert_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alerts", alert_stats.get('total_alerts', 0))
            with col2:
                st.metric("Active Alerts", alert_stats.get('active_alerts', 0))
            with col3:
                st.metric("Resolved Alerts", alert_stats.get('resolved_alerts', 0))
            with col4:
                st.metric("High Priority", alert_stats.get('high_priority_alerts', 0))
            
            # Recent alerts
            st.subheader("Recent Alerts")
            recent_alerts = alert_manager.get_recent_alerts(limit=10)
            
            if recent_alerts:
                alert_data = []
                for alert in recent_alerts:
                    alert_data.append({
                        'Alert ID': alert.alert_id,
                        'Timestamp': alert.timestamp,
                        'Priority': alert.priority,
                        'Status': alert.status,
                        'Fraud Score': f"{alert.fraud_score:.4f}",
                        'Transaction Amount': f"{alert.transaction_data.get('amount', 0):,.2f}"
                    })
                
                alerts_df = pd.DataFrame(alert_data)
                st.dataframe(alerts_df)
            else:
                st.info("No recent alerts found.")
                
        except Exception as e:
            st.error(f"Error retrieving alert information: {str(e)}")
        
        # Alert configuration
        st.subheader("Alert Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            fraud_threshold = st.slider("Fraud Alert Threshold", 0.0, 1.0, 0.5, 0.01)
            high_risk_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.8, 0.01)
        
        with col2:
            enable_email = st.checkbox("Enable Email Alerts", value=False)
            enable_sms = st.checkbox("Enable SMS Alerts", value=False)
        
        if st.button("Update Alert Configuration"):
            try:
                # Update fraud detector thresholds
                fraud_detector = st.session_state.fraud_detector
                fraud_detector.update_thresholds(fraud_threshold, high_risk_threshold)
                
                st.success("Alert configuration updated successfully!")
                
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")


def main():
    """Main function to run the dashboard."""
    dashboard = FraudDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()