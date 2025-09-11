# model_validation.py - Comprehensive Model Validation and Testing
import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our training modules
from enhanced_model_training import AdvancedModelTrainer
from synthetic_data_generator import create_enhanced_dataset

class ModelValidator:
    """
    Comprehensive model validation and testing system
    """
    
    def __init__(self, model_path="models", data_path="enhanced_pause_features.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load trained models"""
        print("📂 Loading trained models...")
        
        model_files = {
            'Random Forest': 'random_forest.joblib',
            'XGBoost': 'xgboost.joblib',
            'LightGBM': 'lightgbm.joblib',
            'Ensemble': 'ensemble_model.joblib'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(self.model_path, filename)
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"   ✅ Loaded {name}")
            else:
                print(f"   ❌ {name} not found at {filepath}")
        
        # Load scaler and label encoder
        scaler_path = os.path.join(self.model_path, "scaler.joblib")
        encoder_path = os.path.join(self.model_path, "label_encoder.joblib")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        print(f"📊 Loaded {len(self.models)} models")
    
    def validate_model_performance(self, model_name, X, y, cv_folds=5):
        """Validate model performance with cross-validation"""
        
        print(f"🔍 Validating {model_name}...")
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Calculate metrics
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        print(f"   Cross-validation accuracy: {mean_score:.3f} ± {std_score:.3f}")
        
        # Individual fold scores
        print(f"   Individual fold scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        return {
            'mean_accuracy': mean_score,
            'std_accuracy': std_score,
            'cv_scores': cv_scores.tolist()
        }
    
    def test_on_holdout_data(self, model_name, X_test, y_test):
        """Test model on holdout data"""
        
        print(f"🧪 Testing {model_name} on holdout data...")
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        print(f"   Test accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-score: {f1:.3f}")
        
        # Classification report
        print(f"\n📋 Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred.tolist()
        }
    
    def compare_models(self, X, y):
        """Compare performance of all models"""
        
        print("🏆 Comparing all models...")
        
        comparison_results = {}
        
        for model_name in self.models.keys():
            print(f"\n--- {model_name} ---")
            results = self.validate_model_performance(model_name, X, y)
            if results:
                comparison_results[model_name] = results
        
        # Find best model
        if comparison_results:
            best_model = max(comparison_results.items(), key=lambda x: x[1]['mean_accuracy'])
            print(f"\n🥇 Best model: {best_model[0]} (accuracy: {best_model[1]['mean_accuracy']:.3f})")
        
        return comparison_results
    
    def analyze_feature_importance(self, model_name='Random Forest'):
        """Analyze feature importance"""
        
        print(f"🔍 Analyzing feature importance for {model_name}...")
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(len(model.feature_importances_))])
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("📊 Top 10 Most Important Features:")
            for i, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print(f"❌ {model_name} does not support feature importance")
            return None
    
    def generate_performance_report(self, results):
        """Generate comprehensive performance report"""
        
        print("\n📊 PERFORMANCE REPORT")
        print("=" * 50)
        
        # Overall statistics
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
            worst_model = min(results.items(), key=lambda x: x[1]['mean_accuracy'])
            
            print(f"🥇 Best Model: {best_model[0]}")
            print(f"   Accuracy: {best_model[1]['mean_accuracy']:.3f} ± {best_model[1]['std_accuracy']:.3f}")
            
            print(f"🥉 Worst Model: {worst_model[0]}")
            print(f"   Accuracy: {worst_model[1]['mean_accuracy']:.3f} ± {worst_model[1]['std_accuracy']:.3f}")
            
            # Accuracy range
            accuracies = [r['mean_accuracy'] for r in results.values()]
            print(f"📈 Accuracy Range: {min(accuracies):.3f} - {max(accuracies):.3f}")
            print(f"📊 Average Accuracy: {np.mean(accuracies):.3f}")
            
            # Check if target accuracy achieved
            target_accuracy = 0.85
            if max(accuracies) >= target_accuracy:
                print(f"✅ Target accuracy ({target_accuracy:.1%}) ACHIEVED!")
            else:
                print(f"❌ Target accuracy ({target_accuracy:.1%}) NOT achieved")
                print(f"   Best achieved: {max(accuracies):.1%}")
                print(f"   Gap: {target_accuracy - max(accuracies):.1%}")
        
        return results
    
    def create_validation_plots(self, results, output_dir="validation_plots"):
        """Create validation plots"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not results:
            print("❌ No results to plot")
            return
        
        # Model comparison plot
        plt.figure(figsize=(12, 6))
        
        model_names = list(results.keys())
        accuracies = [results[name]['mean_accuracy'] for name in model_names]
        errors = [results[name]['std_accuracy'] for name in model_names]
        
        plt.bar(model_names, accuracies, yerr=errors, capsize=5, alpha=0.7)
        plt.axhline(y=0.85, color='r', linestyle='--', label='Target Accuracy (85%)')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Validation plots saved to {output_dir}/")
    
    def run_complete_validation(self, test_size=0.2):
        """Run complete validation pipeline"""
        
        print("🚀 Starting Complete Model Validation")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models found. Please train models first.")
            return None
        
        # Load data
        print("\n📊 Loading data...")
        data = pd.read_csv(self.data_path)
        
        # Prepare features
        feature_cols = [col for col in data.columns if col not in ['label', 'filename', 'is_synthetic', 'data_quality_confidence']]
        X = data[feature_cols].fillna(0)
        y = data['label']
        
        # Encode labels
        if hasattr(self, 'label_encoder'):
            y_encoded = self.label_encoder.transform(y)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoder = le
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        if hasattr(self, 'scaler'):
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scaler = scaler
        
        # Validate models
        print("\n🔍 Validating models...")
        validation_results = self.compare_models(X_train, y_train)
        
        # Test on holdout data
        print("\n🧪 Testing on holdout data...")
        test_results = {}
        for model_name in self.models.keys():
            if model_name in ['SVM', 'Logistic Regression', 'Neural Network']:
                test_results[model_name] = self.test_on_holdout_data(model_name, X_test_scaled, y_test)
            else:
                test_results[model_name] = self.test_on_holdout_data(model_name, X_test, y_test)
        
        # Generate report
        print("\n📊 Generating performance report...")
        self.generate_performance_report(validation_results)
        
        # Create plots
        print("\n📈 Creating validation plots...")
        self.create_validation_plots(validation_results)
        
        # Analyze feature importance
        print("\n🔍 Analyzing feature importance...")
        importance_df = self.analyze_feature_importance('Random Forest')
        
        # Save results
        results_summary = {
            'validation_results': validation_results,
            'test_results': test_results,
            'feature_importance': importance_df.to_dict() if importance_df is not None else None,
            'target_accuracy_achieved': max([r['mean_accuracy'] for r in validation_results.values()]) >= 0.85 if validation_results else False
        }
        
        with open('validation_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("\n💾 Validation results saved to validation_results.json")
        
        return results_summary

def main():
    """Main validation function"""
    
    # Initialize validator
    validator = ModelValidator()
    
    # Run complete validation
    results = validator.run_complete_validation()
    
    if results and results['target_accuracy_achieved']:
        print("\n🎉 SUCCESS! Target accuracy of 85% has been achieved!")
    else:
        print("\n⚠️ Target accuracy not achieved. Consider:")
        print("   - Increasing synthetic data samples")
        print("   - Trying different feature selection methods")
        print("   - Adjusting model hyperparameters")
        print("   - Using ensemble methods")
    
    return results

if __name__ == "__main__":
    results = main()
