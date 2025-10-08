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
        
        # Information about large dataset handling
        with st.expander("ℹ️ Large Dataset Information"):
            st.markdown("""
            **This dashboard is optimized to handle large fraud datasets:**
            - ✅ Can load datasets of any size (including 500MB+ files)
            - ⚡ Uses memory-efficient data types to reduce RAM usage
            - 📊 Provides progress indicators for large file loading
            - 🔍 The "Load Fraud Dataset" button loads the complete fraud detection dataset
            
            **Performance Tips:**
            - Large datasets may take 30-60 seconds to load
            - Visualizations are optimized for performance
            - Some operations may be sampled for very large datasets
            """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your fraud detection dataset in CSV format"
        )
        
        # Dataset loading options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Load Fraud Dataset", help="Load the complete fraud detection dataset (Fraud.csv)"):
                self._load_sample_data()
        
        with col2:
            if st.button("🗑️ Clear Data"):
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
        """Load the actual fraud detection dataset."""
        try:
            with st.spinner("Loading fraud dataset... This may take a moment for large files."):
                # Try to load the actual Fraud.csv dataset first
                fraud_dataset_paths = [
                    "data/raw/Fraud.csv",  # The actual fraud dataset
                    "data/raw/fraud.csv",  # Alternative naming
                    "data/processed/Fraud.csv",
                    "data/raw/fraud_data.csv",
                    "data/processed/fraud_data.csv"
                ]
                
                for path in fraud_dataset_paths:
                    try:
                        # Show progress for large file loading
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🔍 Locating dataset...")
                        progress_bar.progress(10)
                        
                        # Check if file exists and get size
                        import os
                        if not os.path.exists(path):
                            continue
                            
                        file_size = os.path.getsize(path) / 1024**2  # Size in MB
                        status_text.text(f"📁 Found dataset ({file_size:.1f} MB). Loading...")
                        progress_bar.progress(20)
                        
                        # Load the full dataset with optimized dtypes for memory efficiency
                        status_text.text("📊 Reading CSV data...")
                        progress_bar.progress(40)
                        
                        # Load the dataset with automatic type inference
                        # Let pandas infer the best data types automatically
                        st.session_state.data = pd.read_csv(path)
                        
                        # Optimize data types after loading for memory efficiency
                        try:
                            # Only optimize numeric columns that are safe to convert
                            if 'isFraud' in st.session_state.data.columns:
                                st.session_state.data['isFraud'] = st.session_state.data['isFraud'].astype('int8')
                            if 'isFlaggedFraud' in st.session_state.data.columns:
                                st.session_state.data['isFlaggedFraud'] = st.session_state.data['isFlaggedFraud'].astype('int8')
                        except:
                            # If optimization fails, keep original types
                            pass
                        
                        progress_bar.progress(80)
                        status_text.text("✅ Processing dataset information...")
                        
                        # Display dataset info
                        progress_bar.progress(100)
                        status_text.text("🎉 Dataset loaded successfully!")
                        
                        st.success(f"✅ **FRAUD DATASET LOADED SUCCESSFULLY** from {path}!")
                        st.info(f"📊 **Dataset shape:** {st.session_state.data.shape[0]:,} rows × {st.session_state.data.shape[1]} columns")
                        
                        # Show memory usage
                        memory_usage = st.session_state.data.memory_usage(deep=True).sum() / 1024**2
                        st.info(f"💾 **Memory usage:** {memory_usage:.1f} MB")
                        
                        # Show basic info about the dataset
                        fraud_count = st.session_state.data['isFraud'].sum() if 'isFraud' in st.session_state.data.columns else 0
                        fraud_rate = fraud_count / len(st.session_state.data) * 100 if len(st.session_state.data) > 0 else 0
                        st.info(f"🚨 **Fraud transactions:** {fraud_count:,} ({fraud_rate:.3f}%)")
                        
                        # Show column information
                        st.info(f"📋 **Columns:** {', '.join(st.session_state.data.columns.tolist())}")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        return
                        
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        st.warning(f"Could not load {path}: {str(e)}")
                        continue
                
                # If no dataset found, show error and generate sample data
                st.error("❌ Could not find the fraud dataset!")
                st.warning("Expected file: data/raw/Fraud.csv")
                st.info("Generating synthetic sample data instead...")
                
                # Generate synthetic sample data as fallback
                st.session_state.data = self._generate_sample_data()
                st.info("Generated synthetic sample data for demonstration.")
                
        except Exception as e:
            st.error(f"Error loading fraud dataset: {str(e)}")
            # Generate sample data as last resort
            st.session_state.data = self._generate_sample_data()
            st.info("Generated synthetic sample data due to error.")
    
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
            with st.spinner("Processing uploaded file... Please wait for large files."):
                # Read the uploaded file
                bytes_data = uploaded_file.read()
                st.session_state.data = pd.read_csv(io.StringIO(bytes_data.decode('utf-8')))
                
                # Display file info
                st.success(f"✅ File uploaded successfully!")
                st.info(f"📊 Dataset shape: {st.session_state.data.shape[0]:,} rows × {st.session_state.data.shape[1]} columns")
                
                # Show memory usage
                memory_usage = st.session_state.data.memory_usage(deep=True).sum() / 1024**2
                st.info(f"💾 Memory usage: {memory_usage:.1f} MB")
                
                # Show fraud info if available
                if 'isFraud' in st.session_state.data.columns:
                    fraud_count = st.session_state.data['isFraud'].sum()
                    fraud_rate = fraud_count / len(st.session_state.data) * 100
                    st.info(f"🚨 Fraud transactions: {fraud_count:,} ({fraud_rate:.3f}%)")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please make sure the file is a valid CSV format.")
    
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
        
        # Show first few rows
        preview_rows = st.slider("Number of rows to preview", 5, 100, 10)
        st.dataframe(data.head(preview_rows), width='stretch')
        
        # Column information
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),  # Convert to string to avoid Arrow issues
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum(),
                'Unique Values': data.nunique()
            })
            st.dataframe(col_info, width='stretch')
        
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
        
        # For very large datasets, use sampling for visualizations to improve performance
        viz_data = data
        is_large_dataset = len(data) > 100000  # Consider datasets with >100k rows as large
        
        if is_large_dataset:
            sample_size = min(50000, len(data))  # Sample up to 50k rows for visualizations
            viz_data = data.sample(n=sample_size, random_state=42)
            st.info(f"📊 Using a sample of {sample_size:,} rows for visualizations (from {len(data):,} total rows)")
        
        # Transaction type distribution
        if 'type' in data.columns:
            # Use full data for counts (aggregated data is small)
            type_counts = data['type'].value_counts().reset_index()
            type_counts.columns = ['Transaction Type', 'Count']
            fig_type = px.bar(
                type_counts,
                x='Transaction Type', y='Count',
                title="Transaction Type Distribution (Full Dataset)",
                labels={'Transaction Type': 'Transaction Type', 'Count': 'Count'}
            )
            st.plotly_chart(fig_type, width='stretch')
        
        # Amount distribution
        if 'amount' in data.columns:
            title_suffix = f" (Sample of {len(viz_data):,} rows)" if is_large_dataset else ""
            fig_amount = px.histogram(
                viz_data, x='amount',
                title=f"Transaction Amount Distribution{title_suffix}",
                nbins=50
            )
            fig_amount.update_xaxes(type="log", title="Amount (log scale)")
            st.plotly_chart(fig_amount, width='stretch')
        
        # Fraud patterns
        if 'isFraud' in data.columns and 'type' in data.columns:
            # Use full data for fraud patterns (aggregated data is small)
            fraud_by_type = data.groupby(['type', 'isFraud']).size().reset_index(name='count')
            fraud_by_type['fraud_status'] = fraud_by_type['isFraud'].map({0: 'Legitimate', 1: 'Fraud'})
            
            fig_fraud = px.bar(
                fraud_by_type, x='type', y='count', color='fraud_status',
                title="Fraud Distribution by Transaction Type (Full Dataset)",
                color_discrete_map={'Legitimate': 'lightblue', 'Fraud': 'red'}
            )
            st.plotly_chart(fig_fraud, width='stretch')
        
        # Additional large dataset insights
        if is_large_dataset:
            st.subheader("📈 Large Dataset Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                unique_accounts = data['nameOrig'].nunique() if 'nameOrig' in data.columns else 0
                st.metric("Unique Accounts", f"{unique_accounts:,}")
            
            with col2:
                if 'amount' in data.columns:
                    total_volume = data['amount'].sum()
                    st.metric("Total Transaction Volume", f"${total_volume:,.0f}")
            
            with col3:
                time_span = "N/A"
                if 'step' in data.columns:
                    time_span = f"{data['step'].max() - data['step'].min()} steps"
                st.metric("Time Span", time_span)
    
    def _preprocess_data(self):
        """Preprocess the data for model training."""
        try:
            with st.spinner("Preprocessing data..."):
                data = st.session_state.data.copy()
                
                # Ensure data types are compatible before processing
                status_text = st.empty()
                status_text.text("🔧 Preparing data types...")
                
                # Convert any problematic data types to standard types
                for col in data.columns:
                    if data[col].dtype == 'object':
                        # Keep string columns as object
                        continue
                    elif data[col].dtype in ['Int64', 'Int32', 'Int16', 'Int8']:
                        # Convert nullable integers to standard integers
                        data[col] = data[col].astype('float64').fillna(0).astype('int64')
                    elif data[col].dtype in ['Float64', 'Float32']:
                        # Convert nullable floats to standard floats
                        data[col] = data[col].astype('float64')
                
                status_text.text("🧹 Cleaning data...")
                # Clean data
                cleaned_data = self.data_cleaner.clean_data(data)
                
                status_text.text("⚙️ Engineering features...")
                # Feature engineering
                engineered_data = self.feature_engineering.engineer_features(cleaned_data)
                
                status_text.text("🔤 Encoding categorical variables...")
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
                
                status_text.text("✅ Preprocessing completed!")
                status_text.empty()
                
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
            st.error("This might be due to data type incompatibilities. Try loading the dataset again.")
            
            # Provide more detailed error information
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.write("**Possible solutions:**")
                st.write("1. Reload the dataset using the 'Load Fraud Dataset' button")
                st.write("2. Check if the dataset has any corrupted values")
                st.write("3. Try uploading a different dataset file")
    
    def _display_model_training(self):
        """Display model training interface."""
        
        # Memory optimization info for large datasets
        if st.session_state.processed_data is not None:
            data_size = len(st.session_state.processed_data)
            if data_size > 500000:
                st.info(f"🧠 **Large Dataset Detected** ({data_size:,} rows)")
                with st.expander("💡 Memory Optimization Info"):
                    st.markdown("""
                    **For large datasets, the system automatically:**
                    - Uses stratified sampling (200k rows) to maintain fraud distribution
                    - Optimizes model parameters to reduce memory usage
                    - Provides detailed progress and error handling
                    
                    **This ensures reliable training while maintaining model quality.**
                    """)
        
        # Model selection
        st.write("Select models to train:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_lr = st.checkbox("Logistic Regression", value=True, help="Fast, memory-efficient linear model")
        with col2:
            train_rf = st.checkbox("Random Forest", value=True, help="Ensemble model, moderate memory usage")
        with col3:
            train_xgb = st.checkbox("XGBoost", value=True, help="Gradient boosting, higher memory usage")
        
        # Training parameters
        st.subheader("Training Parameters")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Train models button
        if st.button("Train Selected Models"):
            self._train_models(train_lr, train_rf, train_xgb, test_size, random_state)
    
    def _train_models(self, train_lr: bool, train_rf: bool, train_xgb: bool, 
                     test_size: float, random_state: int):
        """Train selected models with memory optimization for large datasets."""
        try:
            data = st.session_state.processed_data
            
            # Prepare features and target
            if 'isFraud' not in data.columns:
                st.error("Target variable 'isFraud' not found in data!")
                return
            
            # Memory optimization for large datasets
            original_size = len(data)
            is_large_dataset = original_size > 500000  # Consider >500k rows as large
            
            if is_large_dataset:
                # Use stratified sampling to maintain fraud ratio while reducing memory usage
                max_training_size = 200000  # Limit to 200k rows for training
                
                st.warning(f"🧠 **Large Dataset Detected** ({original_size:,} rows)")
                st.info(f"📊 Using stratified sampling of {max_training_size:,} rows for model training to optimize memory usage")
                
                # Stratified sampling to maintain fraud distribution
                fraud_data = data[data['isFraud'] == 1]
                normal_data = data[data['isFraud'] == 0]
                
                # Calculate sampling ratios
                fraud_ratio = len(fraud_data) / len(data)
                fraud_sample_size = min(len(fraud_data), int(max_training_size * fraud_ratio))
                normal_sample_size = max_training_size - fraud_sample_size
                
                # Sample data
                fraud_sample = fraud_data.sample(n=fraud_sample_size, random_state=random_state)
                normal_sample = normal_data.sample(n=normal_sample_size, random_state=random_state)
                
                # Combine samples
                data = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=random_state)
                
                st.info(f"🎯 **Sampled Data**: {len(data):,} rows (Fraud: {len(fraud_sample):,}, Normal: {len(normal_sample):,})")
                st.info(f"📈 **Fraud Rate Maintained**: {len(fraud_sample)/len(data):.3%} (Original: {fraud_ratio:.3%})")
            
            # Remove non-numerical columns and target variable
            exclude_cols = ['isFraud', 'nameOrig', 'nameDest']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            # Select only numerical features for training
            X = data[feature_cols].select_dtypes(include=[np.number])
            y = data['isFraud']
            
            st.info(f"🔢 **Features**: Using {len(X.columns)} numerical features")
            st.info(f"📋 **Feature List**: {', '.join(list(X.columns)[:8])}{'...' if len(X.columns) > 8 else ''}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            st.info(f"📊 **Training Set**: {len(X_train):,} samples")
            st.info(f"🧪 **Test Set**: {len(X_test):,} samples")
            
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
            
            # Train models with memory optimization
            successful_models = 0
            for i, (name, model) in enumerate(models_to_train):
                try:
                    status_text.text(f"🤖 Training {name}...")
                    
                    # Configure model for memory efficiency
                    if hasattr(model, 'set_params'):
                        if 'Random Forest' in name:
                            # Limit Random Forest parameters for memory efficiency
                            model.set_params(n_jobs=1, max_depth=10)  # Reduce parallelism and depth
                        elif 'XGBoost' in name:
                            # Limit XGBoost parameters for memory efficiency
                            model.set_params(n_jobs=1, max_depth=6)  # Reduce parallelism and depth
                    
                    # Train model
                    model.train(X_train, y_train)
                    
                    status_text.text(f"📊 Evaluating {name}...")
                    
                    # Evaluate model
                    results = self.model_evaluator.evaluate_model(model, X_test, y_test, name)
                    
                    # Store results
                    st.session_state.trained_models[name] = model
                    st.session_state.model_results[name] = results
                    
                    successful_models += 1
                    st.success(f"✅ {name} trained successfully!")
                    
                except Exception as model_error:
                    st.error(f"❌ Failed to train {name}: {str(model_error)}")
                    if "memory" in str(model_error).lower() or "paging file" in str(model_error).lower():
                        st.warning(f"💾 {name} failed due to memory constraints. Try with a smaller dataset or increase virtual memory.")
                    continue
                
                progress_bar.progress((i + 1) / len(models_to_train))
            
            status_text.text("🎉 Training completed!")
            
            if successful_models > 0:
                st.success(f"🎯 Successfully trained {successful_models}/{len(models_to_train)} models!")
                if is_large_dataset:
                    st.info("💡 **Note**: Models were trained on a representative sample of your large dataset for memory efficiency.")
            else:
                st.error("❌ No models were successfully trained. This may be due to memory constraints.")
                st.info("💡 **Suggestions**: Try reducing the dataset size or increasing your system's virtual memory.")
            
        except Exception as e:
            st.error(f"❌ Error during model training: {str(e)}")
            
            # Provide specific guidance for memory errors
            if "memory" in str(e).lower() or "paging file" in str(e).lower():
                with st.expander("🔧 Memory Error Solutions"):
                    st.markdown("""
                    **This error occurs when processing very large datasets. Try these solutions:**
                    
                    1. **Increase Virtual Memory (Recommended)**:
                       - Go to System Properties → Advanced → Performance Settings → Advanced → Virtual Memory
                       - Set custom size: Initial = 4096 MB, Maximum = 8192 MB or higher
                    
                    2. **Use Smaller Dataset**:
                       - The system automatically samples large datasets, but you can try uploading a smaller file
                    
                    3. **Close Other Applications**:
                       - Free up RAM by closing unnecessary programs
                    
                    4. **Restart the Dashboard**:
                       - Sometimes helps clear memory leaks
                    """)
            else:
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
    
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
        
        st.plotly_chart(fig, width='stretch')
        
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
                    st.plotly_chart(fig_cm, width='stretch')
        
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
        
        st.plotly_chart(fig_roc, width='stretch')
    
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