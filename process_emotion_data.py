import pandas as pd
import numpy as np
import os

def process_deam_emotions():
    print("🎭 Processing DEAM Emotion Dataset for Training")
    print("=" * 50)
    
    # Load the main valence and arousal files
    valence_file = "data/deam/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv"
    arousal_file = "data/deam/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
    
    # Load static annotations (song-level averages)
    static_file1 = "data/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    static_file2 = "data/deam/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"
    
    print("📂 Loading emotion annotation files...")
    
    # Load valence data
    valence_df = pd.read_csv(valence_file)
    print(f"✅ Valence data: {valence_df.shape}")
    
    # Load arousal data  
    arousal_df = pd.read_csv(arousal_file)
    print(f"✅ Arousal data: {arousal_df.shape}")
    
    # Load static annotations
    static_df1 = pd.read_csv(static_file1)
    static_df2 = pd.read_csv(static_file2)
    static_df = pd.concat([static_df1, static_df2], ignore_index=True)
    print(f"✅ Static annotations: {static_df.shape}")
    
    # Process dynamic data to get song-level averages
    print(f"\n📊 Processing dynamic emotion data...")
    
    def calculate_song_emotions(df, emotion_type):
        """Calculate average emotion for each song from time-series data"""
        song_emotions = []
        
        for _, row in df.iterrows():
            song_id = row['song_id']
            
            # Get all emotion values (excluding song_id column)
            emotion_values = row.drop('song_id').values
            
            # Remove NaN values and calculate average
            valid_values = emotion_values[~pd.isna(emotion_values)]
            
            if len(valid_values) > 0:
                avg_emotion = np.mean(valid_values)
                song_emotions.append({
                    'song_id': song_id,
                    f'avg_{emotion_type}': avg_emotion,
                    f'std_{emotion_type}': np.std(valid_values),
                    f'samples_{emotion_type}': len(valid_values)
                })
        
        return pd.DataFrame(song_emotions)
    
    # Calculate averages for valence and arousal
    valence_processed = calculate_song_emotions(valence_df, 'valence')
    arousal_processed = calculate_song_emotions(arousal_df, 'arousal')
    
    print(f"✅ Processed valence: {valence_processed.shape[0]} songs")
    print(f"✅ Processed arousal: {arousal_processed.shape[0]} songs")
    
    # Merge valence and arousal data
    emotion_df = pd.merge(valence_processed, arousal_processed, on='song_id')
    
    print(f"\n🎯 Combined emotion data: {emotion_df.shape}")
    print(f"📊 Columns: {list(emotion_df.columns)}")
    
    # Create emotion quadrants (4-class classification)
    def classify_emotion_quadrant(valence, arousal):
        """Classify into 4 emotion quadrants"""
        if valence >= 0 and arousal >= 0:
            return 'happy_energetic'  # Q1: Happy/Excited
        elif valence < 0 and arousal >= 0:
            return 'angry_energetic'  # Q2: Angry/Tense  
        elif valence < 0 and arousal < 0:
            return 'sad_calm'         # Q3: Sad/Depressed
        else:
            return 'peaceful_calm'    # Q4: Peaceful/Relaxed
    
    emotion_df['emotion_quadrant'] = emotion_df.apply(
        lambda row: classify_emotion_quadrant(row['avg_valence'], row['avg_arousal']), 
        axis=1
    )
    
    # Show quadrant distribution
    print(f"\n🎭 Emotion Quadrant Distribution:")
    quadrant_counts = emotion_df['emotion_quadrant'].value_counts()
    for quadrant, count in quadrant_counts.items():
        percentage = (count / len(emotion_df)) * 100
        print(f"   {quadrant}: {count} songs ({percentage:.1f}%)")
    
    # Create simplified emotion labels (binary and 3-class)
    emotion_df['valence_binary'] = (emotion_df['avg_valence'] >= 0).astype(int)  # 0=negative, 1=positive
    emotion_df['arousal_binary'] = (emotion_df['avg_arousal'] >= 0).astype(int)  # 0=low, 1=high
    
    # 3-class emotion (simple)
    def simple_emotion(valence, arousal):
        if valence >= 0.3:
            return 'happy'
        elif valence <= -0.3:
            return 'sad'
        else:
            return 'neutral'
    
    emotion_df['emotion_simple'] = emotion_df.apply(
        lambda row: simple_emotion(row['avg_valence'], row['avg_arousal']), 
        axis=1
    )
    
    print(f"\n😊 Simple Emotion Distribution:")
    simple_counts = emotion_df['emotion_simple'].value_counts()
    for emotion, count in simple_counts.items():
        percentage = (count / len(emotion_df)) * 100
        print(f"   {emotion}: {count} songs ({percentage:.1f}%)")
    
    # Save processed emotion data
    os.makedirs('data/processed', exist_ok=True)
    emotion_df.to_csv('data/processed/emotion_labels.csv', index=False)
    
    print(f"\n💾 Processed emotion data saved to: data/processed/emotion_labels.csv")
    
    # Show sample data
    print(f"\n📋 Sample processed data:")
    print(emotion_df[['song_id', 'avg_valence', 'avg_arousal', 'emotion_quadrant', 'emotion_simple']].head())
    
    return emotion_df

if __name__ == "__main__":
    emotion_data = process_deam_emotions()
    
    print(f"\n🎉 EMOTION DATA PROCESSING COMPLETE!")
    print(f"🎭 Ready for emotion model training!")
    print(f"📊 {len(emotion_data)} songs with emotion labels")