# Model Evaluation Script for Enhanced Speech Pace Management Model
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Configuration - will be set dynamically based on script location
MODEL_PATH = None
CONFIG_PATH = None
DATA_PATH = None

def load_model_and_data():
    """Load the trained model and configuration"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set paths dynamically
        model_path = os.path.join(script_dir, "enhanced_pause_model.joblib")
        config_path = os.path.join(script_dir, "enhanced_pause_features.json")
        data_path = os.path.join(script_dir, "enhanced_pause_features.csv")
        
        # Load model
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded: {config['model_type']}")
        
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        return model, config, df
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def prepare_features(df, config):
    """Prepare features for evaluation"""
    # Get feature columns (excluding non-feature columns)
    feature_cols = config['feature_order']
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Prepare X and y
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"📊 Features prepared: {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"📊 Class distribution:\n{pd.Series(y).value_counts()}")
    
    return X, y

def evaluate_model_performance(model, X, y, config):
    """Evaluate model performance with comprehensive metrics"""
    print("\n" + "="*60)
    print("🎯 MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Basic predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision_macro = precision_score(y, y_pred, average='macro')
    recall_macro = recall_score(y, y_pred, average='macro')
    f1_macro = f1_score(y, y_pred, average='macro')
    
    # Per-class metrics
    precision_per_class = precision_score(y, y_pred, average=None)
    recall_per_class = recall_score(y, y_pred, average=None)
    f1_per_class = f1_score(y, y_pred, average=None)
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision (Macro):  {precision_macro:.4f}")
    print(f"   Recall (Macro):     {recall_macro:.4f}")
    print(f"   F1-Score (Macro):   {f1_macro:.4f}")
    
    # Per-class performance
    classes = model.classes_
    print(f"\n📊 PER-CLASS PERFORMANCE:")
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 55)
    for i, cls in enumerate(classes):
        print(f"{cls:<25} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f}")
    
    return y_pred, y_pred_proba

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png"):
    """Create and display confusion matrix"""
    print(f"\n🔍 CONFUSION MATRIX:")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Display as text
    print("\nConfusion Matrix (Raw counts):")
    print(f"{'Predicted':>15}", end="")
    for cls in classes:
        print(f"{cls:>15}", end="")
    print()
    
    for i, true_cls in enumerate(classes):
        print(f"{true_cls:>15}", end="")
        for j, pred_cls in enumerate(classes):
            print(f"{cm[i, j]:>15}", end="")
        print()
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    print("\nConfusion Matrix (Percentages):")
    print(f"{'Predicted':>15}", end="")
    for cls in classes:
        print(f"{cls:>15}", end="")
    print()
    
    for i, true_cls in enumerate(classes):
        print(f"{true_cls:>15}", end="")
        for j, pred_cls in enumerate(classes):
            print(f"{cm_percent[i, j]:>14.1f}%", end="")
        print()
    
    # Create visual plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Speech Pace Management Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Confusion matrix plot saved as: {save_path}")
    
    return cm

def detailed_classification_report(y_true, y_pred, classes):
    """Display detailed classification report"""
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print("="*60)
    
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    
    return report

def cross_validation_evaluation(model, X, y, cv_folds=5):
    """Perform cross-validation evaluation"""
    print(f"\n🔄 CROSS-VALIDATION EVALUATION ({cv_folds} folds):")
    print("="*60)
    
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
    cv_recall = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    
    print(f"Cross-Validation Results:")
    print(f"   Accuracy:  {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"   Precision: {cv_precision.mean():.4f} (+/- {cv_precision.std() * 2:.4f})")
    print(f"   Recall:    {cv_recall.mean():.4f} (+/- {cv_recall.std() * 2:.4f})")
    print(f"   F1-Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    
    return {
        'accuracy': cv_accuracy,
        'precision': cv_precision,
        'recall': cv_recall,
        'f1': cv_f1
    }

def feature_importance_analysis(model, config):
    """Analyze feature importance if available"""
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    print("="*60)
    
    if hasattr(model, 'feature_importances_'):
        # Get feature importance
        importance = model.feature_importances_
        feature_names = config['feature_order']
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 20 Most Important Features:")
        print(importance_df.head(20).to_string(index=False))
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Feature importance plot saved as: feature_importance.png")
        
        return importance_df
    else:
        print("❌ Feature importance not available for this model type")
        return None

def save_evaluation_results(results, save_path="evaluation_results.txt"):
    """Save evaluation results to file"""
    with open(save_path, 'w') as f:
        f.write("SPEECH PACE MANAGEMENT MODEL EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model Type: {results['model_type']}\n")
        f.write(f"Dataset Size: {results['dataset_size']}\n")
        f.write(f"Number of Features: {results['num_features']}\n")
        f.write(f"Number of Classes: {results['num_classes']}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision (Macro): {results['precision_macro']:.4f}\n")
        f.write(f"Recall (Macro): {results['recall_macro']:.4f}\n")
        f.write(f"F1-Score (Macro): {results['f1_macro']:.4f}\n\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        for cls, count in results['class_distribution'].items():
            f.write(f"{cls}: {count}\n")
    
    print(f"📄 Evaluation results saved to: {save_path}")

def main():
    """Main evaluation function"""
    print("🚀 SPEECH PACE MANAGEMENT MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model, config, df = load_model_and_data()
    if model is None:
        return
    
    # Prepare features
    X, y = prepare_features(df, config)
    
    # Evaluate model performance
    y_pred, y_pred_proba = evaluate_model_performance(model, X, y, config)
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y, y_pred, model.classes_)
    
    # Detailed classification report
    report = detailed_classification_report(y, y_pred, model.classes_)
    
    # Cross-validation evaluation
    cv_results = cross_validation_evaluation(model, X, y)
    
    # Feature importance analysis
    importance_df = feature_importance_analysis(model, config)
    
    # Save results
    results = {
        'model_type': config['model_type'],
        'dataset_size': len(df),
        'num_features': X.shape[1],
        'num_classes': len(model.classes_),
        'accuracy': accuracy_score(y, y_pred),
        'precision_macro': precision_score(y, y_pred, average='macro'),
        'recall_macro': recall_score(y, y_pred, average='macro'),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'class_distribution': dict(pd.Series(y).value_counts())
    }
    
    save_evaluation_results(results)
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"📊 Results saved to: evaluation_results.txt")
    print(f"📈 Confusion matrix saved to: confusion_matrix.png")
    if importance_df is not None:
        print(f"🔍 Feature importance saved to: feature_importance.png")

if __name__ == "__main__":
    main()
