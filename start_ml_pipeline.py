import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def start_ml_pipeline():
    print("🤖 Starting ML Pipeline for Genre Classification")
    print("=" * 50)
    
    # Load pre-computed features
    features_file = "data/gtzan/Data/features_30_sec.csv"
    df = pd.read_csv(features_file)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📊 Features available: {df.columns.tolist()}")
    
    # Extract genre from filename
    df['genre'] = df['filename'].str.extract('([a-z]+)\\.')[0]
    
    # Check genre distribution
    print(f"\n🎵 Genre distribution:")
    print(df['genre'].value_counts())
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col not in ['filename', 'length', 'genre']]
    X = df[feature_columns].values
    y = df['genre'].values
    
    print(f"\n🔢 Feature matrix shape: {X.shape}")
    print(f"🏷️ Labels shape: {y.shape}")
    print(f"🎯 Unique genres: {len(np.unique(y))}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 Training set: {X_train.shape}")
    print(f"📉 Test set: {X_test.shape}")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"\n✅ Data ready for ML training!")
    print(f"🎯 Next steps: Train your first model")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, le

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_encoder = start_ml_pipeline()