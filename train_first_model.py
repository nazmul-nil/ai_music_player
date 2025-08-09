import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_genre_classifier():
    print("🎵 Training Genre Classification Model")
    print("=" * 50)
    
    # Load and prepare data (same as before)
    features_file = "data/gtzan/Data/features_30_sec.csv"
    df = pd.read_csv(features_file)
    
    # Extract genre from filename
    df['genre'] = df['filename'].str.extract('([a-z]+)\\.')[0]
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns 
                      if col not in ['filename', 'length', 'genre', 'label']]
    X = df[feature_columns].values
    y = df['genre'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"📊 Training data: {X_train_scaled.shape}")
    print(f"🎯 Genres: {list(le.classes_)}")
    
    # Train Multiple Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔄 Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train_encoded)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test_encoded, y_pred)
        results[model_name] = accuracy
        
        print(f"✅ {model_name} Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed results for best model
        if accuracy > 0.8:  # If accuracy is good
            print(f"\n📊 Detailed Results for {model_name}:")
            print(classification_report(y_test_encoded, y_pred, 
                                       target_names=le.classes_))
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_accuracy = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"🎯 Best Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Save the best model
    best_model = models[best_model_name]
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model, scaler, and label encoder
    joblib.dump(best_model, 'models/genre_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print(f"\n💾 Model saved to: models/genre_classifier.pkl")
    print(f"💾 Scaler saved to: models/scaler.pkl")
    print(f"💾 Label encoder saved to: models/label_encoder.pkl")
    
    # Test prediction on single song
    print(f"\n🎵 Testing prediction on first test song...")
    single_prediction = best_model.predict(X_test_scaled[0:1])
    predicted_genre = le.inverse_transform(single_prediction)[0]
    actual_genre = y_test[0]
    
    print(f"   🎯 Predicted: {predicted_genre}")
    print(f"   ✅ Actual: {actual_genre}")
    print(f"   📊 Correct: {'YES' if predicted_genre == actual_genre else 'NO'}")
    
    return best_model, scaler, le, best_accuracy

def test_saved_model():
    """Test loading and using the saved model"""
    print(f"\n🔄 Testing saved model...")
    
    try:
        # Load saved model
        model = joblib.load('models/genre_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le = joblib.load('models/label_encoder.pkl')
        
        print("✅ Model loaded successfully!")
        print("🎯 Ready for mobile deployment!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    model, scaler, label_encoder, accuracy = train_genre_classifier()
    test_saved_model()
    
    print(f"\n🎉 CONGRATULATIONS!")
    print(f"🎵 Your first AI music model is ready!")
    print(f"🎯 Genre classification accuracy: {accuracy*100:.1f}%")
    print(f"📱 Ready for React Native integration!")