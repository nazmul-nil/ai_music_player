import os
import pandas as pd
import numpy as np

def explore_deam_dataset():
    print("🎭 Exploring DEAM Emotion Dataset")
    print("=" * 40)
    
    # Check what we downloaded
    deam_path = "data/deam"
    
    if os.path.exists(deam_path):
        print("📁 DEAM dataset structure:")
        for root, dirs, files in os.walk(deam_path):
            level = root.replace(deam_path, '').count(os.sep)
            indent = "  " * level
            print(f"{indent}📂 {os.path.basename(root)}/")
            
            subindent = "  " * (level + 1)
            for file in files[:5]:  # Show first 5 files
                if file.endswith(('.csv', '.txt', '.json')):
                    print(f"{subindent}📄 {file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
    
    # Look for annotation files
    annotation_files = []
    for root, dirs, files in os.walk(deam_path):
        for file in files:
            if file.endswith('.csv') and ('annotation' in file.lower() or 'emotion' in file.lower() or 'valence' in file.lower()):
                annotation_files.append(os.path.join(root, file))
    
    print(f"\n🎯 Found {len(annotation_files)} annotation files:")
    for file in annotation_files:
        print(f"  📋 {os.path.basename(file)}")
    
    # Try to load and examine the main annotation file
    if annotation_files:
        main_file = annotation_files[0]
        try:
            df = pd.read_csv(main_file)
            print(f"\n📊 Emotion data preview:")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"\n📈 First few rows:")
            print(df.head())
            
            # Check for valence/arousal columns
            valence_cols = [col for col in df.columns if 'valence' in col.lower()]
            arousal_cols = [col for col in df.columns if 'arousal' in col.lower()]
            
            if valence_cols:
                print(f"\n🎭 Valence columns: {valence_cols}")
            if arousal_cols:
                print(f"🎭 Arousal columns: {arousal_cols}")
                
        except Exception as e:
            print(f"❌ Error reading {main_file}: {e}")
    
    else:
        print("❌ No annotation files found. Check dataset structure.")

if __name__ == "__main__":
    explore_deam_dataset()