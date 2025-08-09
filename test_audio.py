import os
import pandas as pd

def test_audio_loading():
    print("🎵 Testing Audio Processing...")
    print("=" * 40)
    
    # Test if we can import required libraries
    try:
        import librosa
        print("✅ librosa imported successfully")
    except ImportError:
        print("❌ librosa not installed. Run: pip install librosa")
        return
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError:
        print("❌ numpy not installed. Run: pip install numpy")
        return
    
    # Test loading one audio file
    test_file = "data/gtzan/Data/genres_original/blues/blues.00000.wav"
    
    if os.path.exists(test_file):
        print(f"\n🎶 Testing audio file: {test_file}")
        
        try:
            # Load audio file
            y, sr = librosa.load(test_file, duration=10)  # Load first 10 seconds
            print(f"✅ Audio loaded successfully!")
            print(f"   📊 Sample rate: {sr} Hz")
            print(f"   📊 Duration: {len(y)/sr:.2f} seconds")
            print(f"   📊 Audio shape: {y.shape}")
            
            # Extract basic features
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            print(f"   🎵 Estimated tempo: {tempo:.1f} BPM")
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
    else:
        print(f"❌ Test file not found: {test_file}")
    
    # Check CSV features
    csv_file = "data/gtzan/Data/features_30_sec.csv"
    if os.path.exists(csv_file):
        print(f"\n📋 Checking pre-computed features...")
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ CSV loaded successfully!")
            print(f"   📊 Shape: {df.shape}")
            print(f"   📊 Columns: {list(df.columns[:5])}... (showing first 5)")
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")

if __name__ == "__main__":
    test_audio_loading()