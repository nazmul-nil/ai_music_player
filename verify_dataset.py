import os

def explore_gtzan_dataset():
    base_path = "data/gtzan"
    
    print("🎵 GTZAN Dataset Structure:")
    print("=" * 40)
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)
        
        print(f"{indent}📁 {folder_name}/")
        
        # Show files (limit to first few)
        subindent = "  " * (level + 1)
        for i, file in enumerate(files[:5]):
            if file.endswith(('.wav', '.au', '.csv')):
                print(f"{subindent}🎶 {file}")
        
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")
    
    # Count genre folders and songs
    genres_path = os.path.join(base_path, "Data", "genres_original")
    if os.path.exists(genres_path):
        print("\n🎯 Genre Analysis:")
        print("=" * 40)
        
        total_songs = 0
        for genre in os.listdir(genres_path):
            genre_path = os.path.join(genres_path, genre)
            if os.path.isdir(genre_path):
                song_files = [f for f in os.listdir(genre_path) 
                             if f.endswith(('.wav', '.au'))]
                song_count = len(song_files)
                total_songs += song_count
                print(f"  {genre}: {song_count} songs")
        
        print(f"\n📊 Total Songs: {total_songs}")
        print(f"📊 Total Genres: {len(os.listdir(genres_path))}")
    
    # Check for CSV files with features
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if csv_files:
        print(f"\n📄 Found {len(csv_files)} CSV files with features:")
        for csv_file in csv_files:
            print(f"  📋 {os.path.basename(csv_file)}")

if __name__ == "__main__":
    explore_gtzan_dataset()