# Disease Outbreak Prediction System for SDG 3: Good Health and Well-Being
# Author: AI/ML Assignment Solution
# Description: Predicts disease outbreak risk using supervised learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DiseaseOutbreakPredictor:
    """
    A machine learning system to predict disease outbreak risk based on 
    environmental, demographic, and health infrastructure factors.
    
    Addresses SDG 3: Good Health and Well-Being by enabling early warning
    systems for disease prevention and response.
    """
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.feature_names = []
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic health data for demonstration purposes.
        In real implementation, use WHO, CDC, or government health datasets.
        """
        print("Generating synthetic health dataset...")
        
        # Environmental factors
        temperature = np.random.normal(25, 10, n_samples)  # Celsius
        humidity = np.random.uniform(30, 90, n_samples)    # Percentage
        rainfall = np.random.exponential(50, n_samples)    # mm per month
        air_quality_index = np.random.uniform(20, 300, n_samples)
        
        # Demographic factors
        population_density = np.random.exponential(100, n_samples)  # per km²
        urbanization_rate = np.random.uniform(20, 90, n_samples)   # Percentage
        poverty_rate = np.random.uniform(5, 60, n_samples)         # Percentage
        
        # Health infrastructure
        hospitals_per_capita = np.random.exponential(0.5, n_samples)
        doctors_per_capita = np.random.exponential(1.0, n_samples)
        vaccination_coverage = np.random.uniform(40, 95, n_samples)  # Percentage
        
        # Historical disease data
        previous_outbreaks = np.random.poisson(2, n_samples)
        seasonal_trend = np.sin(np.random.uniform(0, 2*np.pi, n_samples))
        
        # Create outbreak risk based on realistic factors
        # Higher risk with: high temperature, high humidity, poor air quality,
        # high population density, low vaccination, few hospitals
        risk_score = (
            0.15 * (temperature > 30) +
            0.15 * (humidity > 70) +
            0.20 * (air_quality_index > 150) +
            0.10 * (population_density > 200) +
            0.15 * (vaccination_coverage < 70) +
            0.10 * (hospitals_per_capita < 0.3) +
            0.10 * (poverty_rate > 30) +
            0.05 * (previous_outbreaks > 3)
        )
        
        # Add some randomness and create binary labels
        risk_score += np.random.normal(0, 0.1, n_samples)
        outbreak_risk = (risk_score > 0.5).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'air_quality_index': air_quality_index,
            'population_density': population_density,
            'urbanization_rate': urbanization_rate,
            'poverty_rate': poverty_rate,
            'hospitals_per_capita': hospitals_per_capita,
            'doctors_per_capita': doctors_per_capita,
            'vaccination_coverage': vaccination_coverage,
            'previous_outbreaks': previous_outbreaks,
            'seasonal_trend': seasonal_trend,
            'outbreak_risk': outbreak_risk
        })
        
        return data
    
    def preprocess_data(self, data):
        """Clean and preprocess the dataset"""
        print("Preprocessing data...")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        # Feature engineering
        data['health_infrastructure_score'] = (
            data['hospitals_per_capita'] + data['doctors_per_capita']
        ) / 2
        
        data['environmental_risk_score'] = (
            (data['air_quality_index'] > 150).astype(int) +
            (data['temperature'] > 30).astype(int) +
            (data['humidity'] > 70).astype(int)
        )
        
        # Separate features and target
        X = data.drop('outbreak_risk', axis=1)
        y = data['outbreak_risk']
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple ML models and compare performance"""
        print("Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'y_test': y_test
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
        
        # Select best model based on AUC score
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest performing model: {best_model_name}")
        
        return results, X_test, y_test
    
    def visualize_results(self, results, X_test, y_test):
        """Create visualizations for model performance and insights"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        auc_scores = [results[name]['auc_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance (using Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[0, 1].barh(feature_importance['feature'], feature_importance['importance'])
            axes[0, 1].set_xlabel('Feature Importance')
            axes[0, 1].set_title('Feature Importance (Random Forest)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
        best_predictions = results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        
        # 4. Risk Distribution
        risk_probs = results[best_model_name]['probabilities']
        axes[1, 1].hist(risk_probs, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Outbreak Risk Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Outbreak Risk Predictions')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed classification report
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(y_test, best_predictions))
    
    def predict_outbreak_risk(self, new_data):
        """Predict outbreak risk for new data points"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please run train_models first.")
        
        # Scale data if using logistic regression
        if isinstance(self.best_model, LogisticRegression):
            new_data_scaled = self.scaler.transform(new_data)
            predictions = self.best_model.predict_proba(new_data_scaled)[:, 1]
        else:
            predictions = self.best_model.predict_proba(new_data)[:, 1]
        
        return predictions
    
    def ethical_analysis(self):
        """Provide ethical considerations and bias analysis"""
        print("\n" + "="*60)
        print("ETHICAL CONSIDERATIONS AND BIAS ANALYSIS")
        print("="*60)
        
        ethical_points = [
            "1. DATA BIAS CONCERNS:",
            "   - Synthetic data may not represent real-world complexity",
            "   - Historical health data may reflect systemic inequalities",
            "   - Underrepresentation of marginalized communities",
            "",
            "2. FAIRNESS AND EQUITY:",
            "   - Model should not discriminate based on socioeconomic status",
            "   - Equal access to early warning systems across all populations",
            "   - Consider cultural and linguistic barriers in implementation",
            "",
            "3. PRIVACY AND CONSENT:",
            "   - Health data requires strict privacy protections",
            "   - Transparent data collection and usage policies",
            "   - Individual consent for health monitoring",
            "",
            "4. ACCOUNTABILITY AND TRANSPARENCY:",
            "   - Clear explanation of model decisions for healthcare workers",
            "   - Regular model auditing and bias testing",
            "   - Human oversight in critical health decisions",
            "",
            "5. BENEFICIAL IMPACT:",
            "   - Enables proactive public health interventions",
            "   - Reduces disease burden in vulnerable populations",
            "   - Supports SDG 3: Good Health and Well-Being goals"
        ]
        
        for point in ethical_points:
            print(point)
        
        print("\nRECOMMENDATIONS:")
        recommendations = [
            "- Use real, representative datasets from WHO or health ministries",
            "- Implement fairness metrics and bias detection algorithms",
            "- Engage with public health experts and community stakeholders",
            "- Establish clear governance frameworks for AI in healthcare",
            "- Provide training for healthcare workers on AI system limitations"
        ]
        
        for rec in recommendations:
            print(rec)

# Main execution
def main():
    """Main function to run the disease outbreak prediction system"""
    print("="*60)
    print("DISEASE OUTBREAK PREDICTION SYSTEM")
    print("SDG 3: Good Health and Well-Being")
    print("="*60)
    
    # Initialize predictor
    predictor = DiseaseOutbreakPredictor()
    
    # Generate and preprocess data
    data = predictor.generate_synthetic_data(1500)
    X, y = predictor.preprocess_data(data)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Outbreak cases: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Train models
    results, X_test, y_test = predictor.train_models(X, y)
    
    # Visualize results
    predictor.visualize_results(results, X_test, y_test)
    
    # Example prediction for new location
    print("\n" + "="*40)
    print("EXAMPLE: PREDICTING RISK FOR NEW LOCATION")
    print("="*40)
    
    new_location = pd.DataFrame({
        'temperature': [32.0],
        'humidity': [85.0],
        'rainfall': [45.0],
        'air_quality_index': [180.0],
        'population_density': [250.0],
        'urbanization_rate': [65.0],
        'poverty_rate': [35.0],
        'hospitals_per_capita': [0.2],
        'doctors_per_capita': [0.8],
        'vaccination_coverage': [60.0],
        'previous_outbreaks': [3],
        'seasonal_trend': [0.5],
        'health_infrastructure_score': [0.5],
        'environmental_risk_score': [3]
    })
    
    risk_probability = predictor.predict_outbreak_risk(new_location)
    print(f"Outbreak Risk Probability: {risk_probability[0]:.2f}")
    
    if risk_probability[0] > 0.7:
        print("⚠️  HIGH RISK: Immediate intervention recommended")
    elif risk_probability[0] > 0.4:
        print("⚡ MODERATE RISK: Enhanced monitoring advised")
    else:
        print("✅ LOW RISK: Continue routine surveillance")
    
    # Ethical analysis
    predictor.ethical_analysis()
    
    print("\n" + "="*60)
    print("PROJECT IMPACT ON SDG 3:")
    print("✓ Early warning system for disease outbreaks")
    print("✓ Resource allocation optimization")
    print("✓ Proactive public health interventions")
    print("✓ Reduced morbidity and mortality rates")
    print("✓ Strengthened health systems resilience")
    print("="*60)

if __name__ == "__main__":
    main()