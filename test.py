import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class CarbonEmissionPredictor:
    """XGBoost regression model for predicting carbon emissions"""
    
    def __init__(self, csv_path, force_retrain=False):
        self.csv_path = csv_path
        self.force_retrain = force_retrain
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.results = {}
        self.preprocessor = None
        self.feature_columns = None
        self.model_save_path = "carbon_emission_xgboost_model.pkl"
        
    def load_existing_model(self):
        """Load existing model from pickle file"""
        if os.path.exists(self.model_save_path):
            try:
                with open(self.model_save_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✅ Loaded existing model from {self.model_save_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return False
        else:
            print(f"❌ No existing model found at {self.model_save_path}")
            return False

    def load_and_explore_data(self):
        """Load and explore the dataset"""
        print("Loading and exploring data...")
        self.data = pd.read_csv(self.csv_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print("\nDataset info:")
        print(self.data.info())
        print("\nTarget variable statistics:")
        print(self.data['CarbonEmission'].describe())
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(self.data['CarbonEmission'], bins=50, alpha=0.7)
        plt.title('Carbon Emission Distribution')
        plt.xlabel('Carbon Emission')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(self.data['CarbonEmission'])
        plt.title('Carbon Emission Box Plot')
        plt.ylabel('Carbon Emission')
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("Preprocessing data...")
        
        self.X = self.data.drop('CarbonEmission', axis=1)
        self.y = self.data['CarbonEmission']
        
        def process_list_column(col):
            """Convert string representation of lists to binary features"""
            all_items = set()
            for item_list in col:
                if pd.notna(item_list) and item_list != '[]':
                    items = item_list.strip("[]").replace("'", "").split(", ")
                    all_items.update([item.strip() for item in items if item.strip()])
            
            binary_cols = {}
            for item in sorted(all_items):
                binary_cols[f"{col.name}_{item}"] = col.apply(
                    lambda x: 1 if pd.notna(x) and item in x else 0
                )
            return pd.DataFrame(binary_cols)
        
        recycling_features = process_list_column(self.X['Recycling'])
        cooking_features = process_list_column(self.X['Cooking_With'])
        
        self.X = self.X.drop(['Recycling', 'Cooking_With'], axis=1)
        self.X = pd.concat([self.X, recycling_features, cooking_features], axis=1)
        
        categorical_features = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 
                              'Heating Energy Source', 'Transport', 'Vehicle Type', 
                              'Social Activity', 'Frequency of Traveling by Air', 
                              'Waste Bag Size', 'Energy efficiency']
        
        numerical_features = ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km',
                            'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
                            'How Many New Clothes Monthly', 'How Long Internet Daily Hour']
        
        binary_features = [col for col in self.X.columns 
                          if col.startswith(('Recycling_', 'Cooking_With_'))]
        numerical_features.extend(binary_features)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])
        
        print(f"Features after preprocessing: {len(self.X.columns)}")
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Binary features: {len(binary_features)}")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        
    def create_model(self):
        """Create XGBoost regression model"""
        print("Creating XGBoost regression model...")
        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])
        
    def train_and_evaluate(self):
        """Train and evaluate the XGBoost model"""
        print("Training and evaluating XGBoost model...")
        
        self.model.fit(self.X_train, self.y_train)
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                  cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'cv_rmse': cv_rmse
        }
        
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"CV RMSE: {cv_rmse:.2f}")
    
    def feature_importance_analysis(self):
        """Analyze feature importance for XGBoost model"""
        print("\nPerforming feature importance analysis...")
        
        # Use built-in feature importance for XGBoost
        importances = self.model.named_steps['regressor'].feature_importances_
        
        # Get feature names after preprocessing
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = importances_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances - XGBoost')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print("Top 10 most important features:")
        print(importances_df.head(10))
    
    def visualize_results(self):
        """Visualize model performance"""
        print("\nCreating visualizations for XGBoost model...")
        
        y_pred = self.model.predict(self.X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Carbon Emission')
        plt.ylabel('Predicted Carbon Emission')
        plt.title('Actual vs Predicted Carbon Emission (XGBoost)')
        plt.tight_layout()
        plt.show()
        
        residuals = self.y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Carbon Emission')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot (XGBoost)')
        plt.tight_layout()
        plt.show()
    
    def predict_new_sample(self, sample_data):
        """Make prediction for new sample using the XGBoost model"""
        if self.model is None:
            print("No model trained yet. Please run the complete pipeline first.")
            return None
        
        if isinstance(sample_data, dict):
            sample_df = pd.DataFrame([sample_data])
        else:
            sample_df = sample_data.copy()
        
        try:
            def process_list_column(col):
                """Convert string representation of lists to binary features"""
                all_items = set()
                for item_list in col:
                    if pd.notna(item_list) and item_list != '[]':
                        items = item_list.strip("[]").replace("'", "").split(", ")
                        all_items.update([item.strip() for item in items if item.strip()])
                
                binary_cols = {}
                for item in sorted(all_items):
                    binary_cols[f"{col.name}_{item}"] = col.apply(
                        lambda x: 1 if pd.notna(x) and item in x else 0
                    )
                return pd.DataFrame(binary_cols)
            
            processed_df = sample_df.copy()
            
            if 'Recycling' in processed_df.columns:
                recycling_features = process_list_column(processed_df['Recycling'])
                processed_df = processed_df.drop(['Recycling'], axis=1)
                processed_df = pd.concat([processed_df, recycling_features], axis=1)
            
            if 'Cooking_With' in processed_df.columns:
                cooking_features = process_list_column(processed_df['Cooking_With'])
                processed_df = processed_df.drop(['Cooking_With'], axis=1)
                processed_df = pd.concat([processed_df, cooking_features], axis=1)
            
            if hasattr(self, 'X') and self.X is not None:
                for col in self.X.columns:
                    if col not in processed_df.columns:
                        processed_df[col] = 0
                processed_df = processed_df[self.X.columns]
            
            prediction = self.model.predict(processed_df)
            return prediction[0] if len(prediction) == 1 else prediction
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            print("This might be due to missing features or incorrect data format.")
            print("Please ensure your sample data contains all required features.")
            return None
    
    def save_model(self):
        """Save the XGBoost model to a file"""
        try:
            with open(self.model_save_path, 'wb') as f:
                pickle.dump(self.model, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("=" * 60)
        print("CARBON EMISSION PREDICTION PIPELINE - XGBOOST")
        print("=" * 60)
        
        # Check if model exists and load it (unless force_retrain is True)
        if not self.force_retrain and self.load_existing_model():
            print("🚀 Using existing trained model. Set force_retrain=True to retrain.")
            # Load data for feature preprocessing consistency
            self.data = pd.read_csv(self.csv_path)
            self.preprocess_data()  # Need this for predict_new_sample to work
            return self.model
        
        print("🔄 Training new model...")
        self.load_and_explore_data()
        self.preprocess_data()
        self.split_data()
        self.create_model()
        self.train_and_evaluate()
        self.feature_importance_analysis()
        self.visualize_results()
        
        print("\n" + "💾 Saving XGBoost model...")
        if self.save_model():
            print("✅ XGBoost model saved successfully for future use!")
        else:
            print("❌ Failed to save model.")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("Model: XGBoost")
        print(f"Test R²: {self.results['test_r2']:.4f}")
        print(f"Test RMSE: {self.results['test_rmse']:.2f}")
        print("=" * 60)
        
        return self.model

if __name__ == "__main__":
    # Use existing model by default, set force_retrain=True to retrain
    predictor = CarbonEmissionPredictor('Carbon Emission.csv', force_retrain=False)
    model = predictor.run_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    sample_input = {
        'Body Type': 'normal',
        'Sex': 'male',
        'Diet': 'omnivore',
        'How Often Shower': 'daily',
        'Heating Energy Source': 'electricity',
        'Transport': 'private',
        'Vehicle Type': 'petrol',
        'Social Activity': 'sometimes',
        'Monthly Grocery Bill': 200,
        'Frequency of Traveling by Air': 'rarely',
        'Vehicle Monthly Distance Km': 1000,
        'Waste Bag Size': 'medium',
        'Waste Bag Weekly Count': 3,
        'How Long TV PC Daily Hour': 4,
        'How Many New Clothes Monthly': 2,
        'How Long Internet Daily Hour': 6,
        'Energy efficiency': 'Sometimes',
        'Recycling': "['Paper', 'Plastic']",
        'Cooking_With': "['Stove', 'Oven']"
    }
    
    predicted_emission = predictor.predict_new_sample(sample_input)
    print(f"Predicted Carbon Emission: {predicted_emission:.2f}")