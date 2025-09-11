# Model Accuracy and Loss Curves Script
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded: {config['model_type']}")
        
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded: {df.shape[0]} samples")
        
        return model, config, df
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def prepare_data_for_training(df, config):
    """Prepare data for training and evaluation"""
    # Get feature columns
    feature_cols = config['feature_order']
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Prepare X and y
    X = df[feature_cols].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"📊 Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def plot_learning_curves(model, X_train, y_train, model_name="Model"):
    """Plot learning curves for training and validation"""
    print(f"\n📈 Generating learning curves for {model_name}...")
    
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot training curve
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    # Plot validation curve
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-Validation Accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"learning_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Learning curves saved as: {filename}")
    
    plt.show()
    
    return train_sizes, train_scores, val_scores

def plot_validation_curves(model, X_train, y_train, model_name="Model"):
    """Plot validation curves for hyperparameter tuning"""
    print(f"\n🔍 Generating validation curves for {model_name}...")
    
    if isinstance(model, RandomForestClassifier):
        # For Random Forest, vary n_estimators
        param_range = [10, 50, 100, 200, 300, 400, 500]
        param_name = 'n_estimators'
        
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, 
            param_name=param_name, param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
    elif isinstance(model, GradientBoostingClassifier):
        # For Gradient Boosting, vary learning_rate
        param_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        param_name = 'learning_rate'
        
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, 
            param_name=param_name, param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
    elif isinstance(model, MLPClassifier):
        # For Neural Network, vary hidden_layer_sizes
        param_range = [(50,), (100,), (100, 50), (200, 100), (200, 100, 50)]
        param_name = 'hidden_layer_sizes'
        
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, 
            param_name=param_name, param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
    else:
        print(f"⚠️ Validation curves not implemented for {type(model).__name__}")
        return None, None
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot training curve
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    # Plot validation curve
    plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-Validation Accuracy')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel('Accuracy')
    plt.title(f'Validation Curves - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"validation_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Validation curves saved as: {filename}")
    
    plt.show()
    
    return param_range, train_scores, val_scores

def retrain_with_progress_tracking(X_train, X_test, y_train, y_test, feature_cols):
    """Retrain models with progress tracking to generate curves"""
    print(f"\n🔄 Retraining models with progress tracking...")
    
    # Initialize models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Training {name}...")
        
        try:
            # Generate learning curves
            train_sizes, train_scores, val_scores = plot_learning_curves(
                model, X_train, y_train, name
            )
            
            # Generate validation curves
            param_range, train_scores_val, val_scores_val = plot_validation_curves(
                model, X_train, y_train, name
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate loss (for classification, use log loss)
            try:
                y_pred_proba = model.predict_proba(X_test)
                loss = log_loss(y_test, y_pred_proba)
            except:
                loss = None
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'loss': loss,
                'learning_curves': (train_sizes, train_scores, val_scores),
                'validation_curves': (param_range, train_scores_val, val_scores_val) if param_range else None
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f if loss else 'N/A'}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue
    
    return results

def plot_model_comparison(results):
    """Plot comparison of all models"""
    if not results:
        return
    
    print(f"\n📊 Creating model comparison plots...")
    
    # Accuracy comparison
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    losses = [results[name]['loss'] for name in names if results[name]['loss'] is not None]
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(names, accuracies, color=['blue', 'green', 'orange'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Loss comparison (if available)
    if losses:
        valid_names = [name for name in names if results[name]['loss'] is not None]
        bars2 = ax2.bar(valid_names, losses, color=['red', 'purple', 'brown'])
        ax2.set_ylabel('Log Loss')
        ax2.set_title('Model Loss Comparison')
        
        # Add value labels on bars
        for bar, loss in zip(bars2, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'Loss not available\nfor all models', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Model Loss Comparison')
    
    plt.tight_layout()
    
    # Save plot
    filename = "model_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Model comparison saved as: {filename}")
    
    plt.show()

def main():
    """Main function to generate all curves"""
    print("🚀 MODEL ACCURACY AND LOSS CURVES GENERATION")
    print("=" * 60)
    
    # Load model and data
    model, config, df = load_model_and_data()
    if model is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_training(df, config)
    
    # Generate curves for existing model
    print(f"\n📈 Generating curves for existing {config['model_type']} model...")
    
    # For existing model, we can only show current performance
    # To get training curves, we need to retrain
    print(f"⚠️ Note: To see training curves, we need to retrain the model.")
    print(f"   The current model was already trained and saved.")
    
    # Retrain models with progress tracking
    results = retrain_with_progress_tracking(X_train, X_test, y_train, y_test, feature_cols)
    
    # Plot model comparison
    plot_model_comparison(results)
    
    print(f"\n🎉 All curves generated successfully!")
    print(f"📊 Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()

