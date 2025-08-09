import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def improve_genre_classifier():
    print("🔧 Improving Genre Classification Model")
    print("=" * 50)
    
    # Load and prepare data
    features_file = "data/gtzan/Data/features_30_sec.csv"
    df = pd.read_csv(features_file)
    df['genre'] = df['filename'].str.extract('([a-z]+)\\.')[0]
    
    # Feature selection - remove less important features
    important_features = [
        'tempo', 'spectral_centroid_mean', 'spectral_bandwidth_mean',
        'rolloff_mean', 'zero_crossing_rate_mean',
        'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean',
        'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean',
        'chroma_stft_mean', 'rms_mean', 'harmony_mean'
    ]
    
    # Add variance features for key MFCCs
    important_features.extend([
        'mfcc1_var', 'mfcc2_var', 'mfcc3_var', 'mfcc4_var'
    ])
    
    X = df[important_features].values
    y = df['genre'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"📊 Using {len(important_features)} selected features")
    print(f"🎯 Feature list: {important_features[:5]}... (+{len(important_features)-5} more)")
    
    # Optimized models with better hyperparameters
    models = {
        'Optimized Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    for model_name, model in models.items():
        print(f"\n🔄 Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train_encoded)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        results[model_name] = accuracy
        
        print(f"✅ {model_name} Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name
    
    print(f"\n🏆 BEST IMPROVED MODEL: {best_model_name}")
    print(f"🎯 Best Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Show improvement
    original_accuracy = 0.71  # From previous run
    improvement = best_accuracy - original_accuracy
    print(f"📈 Improvement: +{improvement:.3f} (+{improvement*100:.1f}%)")
    
    # Detailed results for best model
    y_pred_best = best_model.predict(X_test_scaled)
    print(f"\n📊 Detailed Results:")
    print(classification_report(y_test_encoded, y_pred_best, 
                               target_names=le.classes_))
    
    # Save improved model
    joblib.dump(best_model, 'models/improved_genre_classifier.pkl')
    joblib.dump(scaler, 'models/improved_scaler.pkl')
    joblib.dump(le, 'models/improved_label_encoder.pkl')
    
    print(f"\n💾 Improved model saved!")
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in best_model_name:
        feature_importance = best_model.feature_importances_
        feature_df = pd.DataFrame({
            'feature': important_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Most Important Features:")
        for i, row in feature_df.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
    
    return best_model, best_accuracy

if __name__ == "__main__":
    model, accuracy = improve_genre_classifier()
    
    print(f"\n🎊 MODEL IMPROVEMENT COMPLETE!")
    print(f"🎵 New accuracy: {accuracy*100:.1f}%")
    
    if accuracy > 0.75:
        print(f"🚀 EXCELLENT! Ready for production!")
    elif accuracy > 0.70:
        print(f"✅ GOOD! Suitable for MVP!")
    else:
        print(f"📈 Keep improving...")