\# 🎵 AI Music Player - Dataset Collection



This repository contains the data collection and preprocessing pipeline for our AI-powered music player with on-device machine learning capabilities.



\## 📊 Dataset Overview



We collect multiple datasets to train different AI models:

\- \*\*Genre Classification\*\*: 10,000+ labeled songs across various genres

\- \*\*Mood Detection\*\*: 5,000+ songs with emotion annotations

\- \*\*Audio Features\*\*: Large catalog for recommendation algorithms



---



\## 🎯 1. Genre Classification Datasets



\### GTZAN Dataset - Genre Classification

\*\*Content\*\*: 1,000 tracks of 30-second length across 10 genres (100 tracks each)

\- \*\*Format\*\*: 22050Hz Mono 16-bit WAV files

\- \*\*Genres\*\*: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

\- \*\*Size\*\*: ~1.2GB



\*\*Download Links:\*\*

\- \[Kaggle (Recommended)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

\- \[Alternative Kaggle](https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection)

\- \[Direct Download](http://opihi.cs.uvic.ca/sound/genres.tar.gz)



\### Free Music Archive (FMA) - Large Genre Dataset

\*\*Content\*\*: 106,574 tracks from 16,341 artists across 161 genres

\- \*\*Small Subset\*\*: 8,000 tracks (1,000 per genre) - \*Recommended for initial training\*

\- \*\*Medium Subset\*\*: 25,000 tracks

\- \*\*Large Subset\*\*: 106,574 tracks



\*\*Download Links:\*\*

\- \[GitHub Repository](https://github.com/mdeff/fma)

\- \[Kaggle Alternative](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium)



---



\## 🎭 2. Mood Detection Datasets



\### EMOPIA Dataset - Emotion Recognition

\*\*Content\*\*: 1,087 music clips from 387 songs with emotion labels

\- \*\*Annotations\*\*: Valence-Arousal emotion space (4 quadrants)

\- \*\*Annotators\*\*: 4 dedicated music experts

\- \*\*Size\*\*: ~3GB



\*\*Download Link:\*\*

\- \[Official Website](https://annahung31.github.io/EMOPIA/)



\### DEAM Dataset - Emotional Analysis in Music

\*\*Content\*\*: Music tracks with valence-arousal emotional annotations

\- \*\*Model\*\*: Dimensional emotion model

\- \*\*Size\*\*: ~2GB



\*\*Download Link:\*\*

\- \[Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music)



\### MERP Dataset - Music Emotion Recognition

\*\*Content\*\*: 54 full-length songs with user profile information

\- \*\*Features\*\*: Music features + annotator demographics

\- \*\*Use Case\*\*: Personalized emotion recognition



\*\*Access\*\*: Available through research papers and PMC



\### Memo2496 - Large Emotion Dataset

\*\*Content\*\*: 2,496 instrumental pieces with VA labels and acoustic features

\- \*\*Annotations\*\*: Valence-Arousal labels

\- \*\*Features\*\*: Pre-computed acoustic features included

\- \*\*Size\*\*: ~4GB



\*\*Download Link:\*\*

\- \[FigShare](https://figshare.com/articles/dataset/Memo2496/25827034)



---



\## 🎼 3. Audio Features \& Similarity Training



\### Million Song Dataset

\*\*Content\*\*: Audio features and metadata for 1 million tracks

\- \*\*Note\*\*: ⚠️ Contains features only, no raw audio files

\- \*\*Features\*\*: Echo Nest audio analysis features

\- \*\*Use Case\*\*: Recommendation algorithms, similarity training



\*\*Download Link:\*\*

\- \[Official Website](http://millionsongdataset.com/)



\### Spotify Million Playlist Dataset

\*\*Content\*\*: 1,000,000 playlists with track and playlist metadata

\- \*\*Tracks\*\*: 2+ million unique tracks

\- \*\*Artists\*\*: ~300,000 artists

\- \*\*Use Case\*\*: Playlist generation, music recommendation



\*\*Access Information:\*\*

\- \[Spotify Research](https://research.atspotify.com/datasets/)

\- \*\*Note\*\*: ⚠️ Dataset no longer publicly available - contact Spotify Research directly



---



\## 📋 Quick Start Checklist



\### Initial Setup (Week 1)

\- \[ ] Download GTZAN dataset (1.2GB)

\- \[ ] Set up data processing pipeline

\- \[ ] Test audio feature extraction

\- \[ ] Validate genre classification data



\### Expansion (Week 2+)

\- \[ ] Add FMA Small dataset (7GB)

\- \[ ] Download DEAM for mood detection (2GB)

\- \[ ] Implement emotion recognition pipeline

\- \[ ] Add EMOPIA dataset (3GB)



\### Advanced (Week 3+)

\- \[ ] Integrate Million Song Dataset features

\- \[ ] Contact Spotify Research for playlist data

\- \[ ] Add Memo2496 for enhanced emotion detection



---



\## 🗂️ Recommended Folder Structure



```

data/

├── raw/

│   ├── gtzan/

│   ├── fma/

│   ├── deam/

│   ├── emopia/

│   └── memo2496/

├── processed/

│   ├── features/

│   ├── labels/

│   └── splits/

└── models/

&nbsp;   ├── genre\_classification/

&nbsp;   ├── mood\_detection/

&nbsp;   └── recommendation/

```



---



\## 📝 Notes



\- \*\*Storage Requirements\*\*: ~20GB for all datasets

\- \*\*Processing Time\*\*: 2-8 hours depending on dataset size

\- \*\*Legal\*\*: All datasets are for research purposes - check individual licenses

\- \*\*Priority\*\*: Start with GTZAN → FMA Small → DEAM → Others



---



\## 🤝 Contributing



This is part of our AI Music Player project. For questions about data processing or additional datasets, please create an issue.



---



\*Last Updated: January 2025\*

