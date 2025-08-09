import joblib
import os

def check_mobile_readiness():
    print("📱 Mobile Deployment Readiness Check")
    print("=" * 40)
    
    model_files = [
        'models/improved_genre_classifier.pkl',
        'models/improved_scaler.pkl', 
        'models/improved_label_encoder.pkl'
    ]
    
    total_size = 0
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024 / 1024  # MB
            total_size += size
            print(f"📄 {os.path.basename(file_path)}: {size:.2f} MB")
    
    print(f"\n📊 Total model size: {total_size:.2f} MB")
    
    if total_size < 10:
        print("✅ EXCELLENT for mobile! (<10MB)")
    elif total_size < 25:
        print("✅ GOOD for mobile! (<25MB)")
    else:
        print("⚠️ Consider model compression")
    
    # Test inference speed
    import time
    model = joblib.load('models/improved_genre_classifier.pkl')
    scaler = joblib.load('models/improved_scaler.pkl')
    
    # Simulate mobile inference
    dummy_features = [[0] * 20]  # Adjust size as needed
    
    start_time = time.time()
    for _ in range(100):  # 100 predictions
        prediction = model.predict(scaler.transform(dummy_features))
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # ms
    print(f"⚡ Inference speed: {avg_time:.2f} ms per prediction")
    
    if avg_time < 50:
        print("✅ FAST enough for real-time!")
    elif avg_time < 100:
        print("✅ GOOD for mobile apps!")
    else:
        print("⚠️ Consider optimization")

if __name__ == "__main__":
    check_mobile_readiness()