"""
ETL Pipeline for Data Preprocessing, Transformation, and Loading
Author: Your Name
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('etl_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ETLPipeline:
    """
    Automated ETL Pipeline for data preprocessing and transformation
    """
    
    def __init__(self, config=None):
        """
        Initialize ETL Pipeline with configuration
        
        Args:
            config (dict): Configuration parameters for the pipeline
        """
        self.config = config or {}
        self.data = None
        self.processed_data = None
        self.scaler = None
        self.label_encoders = {}
        self.metadata = {}
        self.data_quality_score = {}
        self.validation_rules = []
        
    # ==================== EXTRACT ====================
    
    def extract_csv(self, filepath, **kwargs):
        """Extract data from CSV file"""
        logger.info(f"Extracting data from CSV: {filepath}")
        try:
            self.data = pd.read_csv(filepath, **kwargs)
            logger.info(f"Successfully loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def extract_json(self, filepath, **kwargs):
        """Extract data from JSON file"""
        logger.info(f"Extracting data from JSON: {filepath}")
        try:
            self.data = pd.read_json(filepath, **kwargs)
            logger.info(f"Successfully loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def extract_excel(self, filepath, sheet_name=0, **kwargs):
        """Extract data from Excel file"""
        logger.info(f"Extracting data from Excel: {filepath}")
        try:
            self.data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    def extract_from_dataframe(self, df):
        """Extract data from existing DataFrame"""
        logger.info("Loading data from DataFrame")
        self.data = df.copy()
        logger.info(f"Successfully loaded {len(self.data)} rows")
        return self.data
    
    # ==================== TRANSFORM ====================
    
    def get_data_profile(self):
        """Generate data profile and statistics"""
        if self.data is None:
            logger.warning("No data loaded")
            return None
        
        profile = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        logger.info("Data Profile Generated:")
        logger.info(f"  Shape: {profile['shape']}")
        logger.info(f"  Duplicates: {profile['duplicates']}")
        logger.info(f"  Memory Usage: {profile['memory_usage']:.2f} MB")
        
        return profile
    
    def handle_missing_values(self, strategy='mean', fill_value=None):
        """
        Handle missing values in the dataset
        
        Args:
            strategy (str): 'mean', 'median', 'most_frequent', 'constant'
            fill_value: Value to use when strategy is 'constant'
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        if self.data is None:
            raise ValueError("No data loaded")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'constant' and fill_value is not None:
                self.data[numeric_cols] = self.data[numeric_cols].fillna(fill_value)
            else:
                imputer = SimpleImputer(strategy=strategy)
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            cat_strategy = 'most_frequent' if strategy != 'constant' else 'constant'
            imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
            self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
        
        logger.info("Missing values handled successfully")
        return self.data
    
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        logger.info("Removing duplicate rows")
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        removed = initial_count - len(self.data)
        logger.info(f"Removed {removed} duplicate rows")
        return self.data
    
    def handle_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Handle outliers using IQR or Z-score method
        
        Args:
            columns (list): Columns to check for outliers
            method (str): 'iqr' or 'zscore'
            threshold (float): IQR multiplier or Z-score threshold
        """
        logger.info(f"Handling outliers using {method} method")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = (z_scores > threshold).sum()
                self.data.loc[z_scores > threshold, col] = self.data[col].median()
            
            logger.info(f"  {col}: {outliers} outliers handled")
        
        return self.data
    
    def encode_categorical(self, columns=None, method='label'):
        """
        Encode categorical variables
        
        Args:
            columns (list): Columns to encode
            method (str): 'label' or 'onehot'
        """
        logger.info(f"Encoding categorical variables using {method} encoding")
        
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns
        
        for col in columns:
            if method == 'label':
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  {col}: Label encoded ({len(le.classes_)} classes)")
                
            elif method == 'onehot':
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)
                logger.info(f"  {col}: One-hot encoded ({dummies.shape[1]} features)")
        
        return self.data
    
    def scale_features(self, columns=None, method='standard'):
        """
        Scale numerical features
        
        Args:
            columns (list): Columns to scale
            method (str): 'standard' or 'minmax'
        """
        logger.info(f"Scaling features using {method} scaling")
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        logger.info(f"Scaled {len(columns)} features")
        
        return self.data
    
    def create_features(self, operations):
        """
        Create new features based on operations
        
        Args:
            operations (dict): Dictionary of new feature names and their operations
                Example: {'total': lambda df: df['col1'] + df['col2']}
        """
        logger.info("Creating new features")
        
        for feature_name, operation in operations.items():
            self.data[feature_name] = operation(self.data)
            logger.info(f"  Created feature: {feature_name}")
        
        return self.data
    
    def select_features(self, features):
        """Select specific features"""
        logger.info(f"Selecting {len(features)} features")
        self.data = self.data[features]
        return self.data
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            target_column (str): Name of the target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed
        """
        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"  Train set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    # ==================== LOAD ====================
    
    def load_to_csv(self, filepath, index=False):
        """Save processed data to CSV"""
        logger.info(f"Loading data to CSV: {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, index=index)
        logger.info("Data saved successfully")
    
    def load_to_json(self, filepath, orient='records'):
        """Save processed data to JSON"""
        logger.info(f"Loading data to JSON: {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_json(filepath, orient=orient, indent=2)
        logger.info("Data saved successfully")
    
    def load_to_excel(self, filepath, sheet_name='Sheet1', index=False):
        """Save processed data to Excel"""
        logger.info(f"Loading data to Excel: {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.data.to_excel(filepath, sheet_name=sheet_name, index=index)
        logger.info("Data saved successfully")
    
    def save_metadata(self, filepath='metadata.json'):
        """Save pipeline metadata"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape if self.data is not None else None,
            'columns': list(self.data.columns) if self.data is not None else None,
            'label_encoders': {k: v.classes_.tolist() for k, v in self.label_encoders.items()},
            'config': self.config,
            'data_quality_score': self.data_quality_score
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {filepath}")

    def add_validation_rule(self, column, rule_type, parameters):
        """
        Add data validation rule
        
        Args:
            column (str): Column to validate
            rule_type (str): Type of validation ('range', 'regex', 'unique', 'relationship')
            parameters (dict): Rule parameters
        """
        self.validation_rules.append({
            'column': column,
            'type': rule_type,
            'parameters': parameters
        })
        
    def validate_data(self):
        """Run all validation rules on the data"""
        logger.info("Validating data against rules")
        validation_results = {}
        
        for rule in self.validation_rules:
            column = rule['column']
            rule_type = rule['type']
            params = rule['parameters']
            
            if rule_type == 'range':
                valid = self.data[column].between(params['min'], params['max'])
            elif rule_type == 'regex':
                valid = self.data[column].str.match(params['pattern'])
            elif rule_type == 'unique':
                valid = ~self.data[column].duplicated()
            elif rule_type == 'relationship':
                valid = self.data[column].isin(params['valid_values'])
            
            validation_results[column] = {
                'pass_rate': valid.mean() * 100,
                'failed_rows': (~valid).sum()
            }
            
            logger.info(f"Validation {column}: {validation_results[column]}")
        
        return validation_results

    def calculate_data_quality_score(self):
        """Calculate overall data quality score"""
        if self.data is None:
            return
        
            # Calculate quality metrics for each column
            self.data_quality_score = {
                'completeness': {},
                'uniqueness': {},
                'validity': {}
            }
        
            for col in self.data.columns:
                self.data_quality_score['completeness'][col] = float(1 - self.data[col].isnull().mean())
                self.data_quality_score['uniqueness'][col] = float(1 - self.data[col].duplicated().mean())
                self.data_quality_score['validity'][col] = float(self.data[col].notna().mean())
        
            # Calculate overall score (0-100)
            metric_means = [
                np.mean(list(metric_scores.values())) 
                for metric_scores in self.data_quality_score.values()
            ]
            self.data_quality_score['overall'] = int(np.mean(metric_means) * 100)
        
            logger.info(f"Data Quality Score: {self.data_quality_score['overall']}/100")
        
            # Log detailed metrics
            for metric, scores in self.data_quality_score.items():
                if metric != 'overall':
                    logger.info(f"{metric.capitalize()} scores by column:")
                    for col, score in scores.items():
                        logger.info(f"  {col}: {score:.2%}")
        
        return self.data_quality_score

    def extract_time_features(self, date_columns):
        """
        Extract time-based features from date columns
        
        Args:
            date_columns (list): List of columns containing dates
        """
        logger.info("Extracting time-based features")
        
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
            
            # Extract basic time components
            self.data[f"{col}_year"] = self.data[col].dt.year
            self.data[f"{col}_month"] = self.data[col].dt.month
            self.data[f"{col}_day"] = self.data[col].dt.day
            self.data[f"{col}_dayofweek"] = self.data[col].dt.dayofweek
            
            # Extract more complex features
            self.data[f"{col}_quarter"] = self.data[col].dt.quarter
            self.data[f"{col}_is_weekend"] = self.data[col].dt.dayofweek.isin([5, 6]).astype(int)
            self.data[f"{col}_is_month_start"] = self.data[col].dt.is_month_start.astype(int)
            self.data[f"{col}_is_month_end"] = self.data[col].dt.is_month_end.astype(int)
            
            # Drop original date column if requested
            if self.config.get('drop_original_date_columns', False):
                self.data = self.data.drop(col, axis=1)
        
        logger.info(f"Created time features for {len(date_columns)} columns")
        return self.data

    def bin_numeric_features(self, columns, n_bins=5, strategy='quantile'):
        """
        Bin numeric features into categories
        
        Args:
            columns (list): Columns to bin
            n_bins (int): Number of bins
            strategy (str): 'quantile' or 'uniform'
        """
        logger.info(f"Binning numeric features using {strategy} strategy")
        
        for col in columns:
            if strategy == 'quantile':
                bins = pd.qcut(self.data[col], n_bins, duplicates='drop')
            else:
                bins = pd.cut(self.data[col], n_bins)
            
            self.data[f"{col}_binned"] = bins
            # Convert to string to avoid JSON serialization issues
            self.data[f"{col}_binned"] = self.data[f"{col}_binned"].astype(str)
            
        return self.data

    # ==================== PIPELINE ====================
    
    def run_pipeline(self, steps):
        """
        Run the complete ETL pipeline
        
        Args:
            steps (list): List of tuples (method_name, kwargs)
        """
        logger.info("=" * 50)
        logger.info("Starting ETL Pipeline")
        logger.info("=" * 50)
        
        for i, (method_name, kwargs) in enumerate(steps, 1):
            logger.info(f"\nStep {i}: {method_name}")
            method = getattr(self, method_name)
            method(**kwargs)
        
        logger.info("\n" + "=" * 50)
        logger.info("ETL Pipeline Completed Successfully")
        logger.info("=" * 50)
        
        return self.data


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Create necessary directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = ETLPipeline()
    
    # Create a more comprehensive test dataset with various data quality issues
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample data with various issues
    sample_data = pd.DataFrame({
        # Numeric columns with missing values and outliers
        'age': np.random.normal(35, 10, n_samples).round(),
        'salary': np.random.normal(60000, 15000, n_samples).round(),
        'experience': np.random.normal(8, 4, n_samples).round(),
        
        # Categorical columns with multiple categories
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_samples),
        'education': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n_samples),
        'performance': np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], n_samples),
        
        # Boolean column
        'remote_work': np.random.choice([True, False], n_samples),
        
        # Date column
        'join_date': [datetime.now() - pd.Timedelta(days=np.random.randint(0, 3650)) for _ in range(n_samples)]
    })
    
    # Introduce missing values (approximately 10% of data)
    for col in sample_data.columns:
        mask = np.random.choice([True, False], size=n_samples, p=[0.1, 0.9])
        sample_data.loc[mask, col] = np.nan
    
    # Add some duplicate rows (approximately 5% of data)
    duplicates = sample_data.sample(n=int(n_samples * 0.05))
    sample_data = pd.concat([sample_data, duplicates], ignore_index=True)
    
    # Add some outliers to numeric columns
    outlier_indices = np.random.choice(sample_data.index, size=int(n_samples * 0.05))
    sample_data.loc[outlier_indices, 'salary'] *= 3  # Triple some salaries
    sample_data.loc[outlier_indices, 'age'] += 30    # Add 30 years to some ages
    
    # Save sample data
    sample_data.to_csv('data/raw/sample_data.csv', index=False)
    
    # Add data validation rules
    pipeline.add_validation_rule('age', 'range', {'min': 18, 'max': 100})
    pipeline.add_validation_rule('salary', 'range', {'min': 20000, 'max': 500000})
    pipeline.add_validation_rule('department', 'relationship', 
                               {'valid_values': ['IT', 'HR', 'Finance', 'Marketing', 'Sales']})
    
    # Define comprehensive pipeline steps
    pipeline_steps = [
        ('extract_csv', {'filepath': 'data/raw/sample_data.csv'}),
        ('get_data_profile', {}),
        
        # Initial data validation
        ('validate_data', {}),
        ('calculate_data_quality_score', {}),
        
        # Data cleaning steps
        ('handle_missing_values', {
            'strategy': 'mean'
        }),
        ('remove_duplicates', {
            'keep': 'first'
        }),
        
        # Time-based feature extraction
        ('extract_time_features', {
            'date_columns': ['join_date']
        }),
        
        # Feature engineering
        ('create_features', {
            'operations': {
                'experience_to_salary_ratio': lambda df: df['salary'] / (df['experience'] + 1),
                'is_senior': lambda df: (df['experience'] >= 10).astype(int),
                'salary_above_avg': lambda df: (df['salary'] > df['salary'].mean()).astype(int),
                'age_group': lambda df: pd.qcut(df['age'], q=4, labels=['Junior', 'Early-Mid', 'Late-Mid', 'Senior']).astype(str)
            }
        }),
        
        # Binning numeric features
        ('bin_numeric_features', {
            'columns': ['salary', 'experience'],
            'n_bins': 5,
            'strategy': 'quantile'
        }),
        
        # Data transformation
        ('encode_categorical', {
            'columns': ['department', 'performance', 'education', 'age_group'],
            'method': 'label'
        }),
        ('handle_outliers', {
            'method': 'iqr',
            'threshold': 1.5,
            'columns': ['age', 'salary', 'experience']
        }),
        ('scale_features', {
            'method': 'standard',
            'columns': ['age', 'salary', 'experience', 'experience_to_salary_ratio']
        }),
        
        # Final data validation
        ('validate_data', {}),
        ('calculate_data_quality_score', {}),
        
        # Data export
        ('load_to_csv', {
            'filepath': 'data/processed/processed_data.csv',
            'index': False
        }),
        ('save_metadata', {
            'filepath': 'data/processed/metadata.json'
        })
    ]
    
    # Run pipeline
    processed_data = pipeline.run_pipeline(pipeline_steps)
    
    print("\n" + "=" * 50)
    print("Processed Data Preview:")
    print("=" * 50)
    print(processed_data.head())
    print(f"\nShape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")