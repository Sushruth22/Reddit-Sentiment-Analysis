# Reddit-Sentiment-Analysis
A comprehensive sentiment analysis application that uses the ELECTRA transformer model to analyze Reddit comments and posts in real-time. The project includes data scraping, preprocessing, model training, and interactive visualization components.

## 🚀 Project Overview

This project implements a complete pipeline for:
- **Real-time Reddit data scraping** using PRAW (Python Reddit API Wrapper)
- **Text preprocessing** with spaCy and NLTK
- **Sentiment classification** using a fine-tuned ELECTRA model
- **Interactive web interface** built with Streamlit
- **Data visualization** with matplotlib and seaborn
- **Named Entity Recognition** for extracting important entities

## 📁 Project Structure

```
Electra/
├── Main.py                          # Main Streamlit application
├── RedditScrape_draft4.py          # Reddit data scraping utilities
├── elect_draft4.py                 # ELECTRA model training script
├── df.csv                          # Main dataset (13MB)
├── df_all.csv                      # Extended dataset (13MB)
├── cleaned_reddit_data.csv         # Processed Reddit data
├── raw_reddit_data.csv             # Raw Reddit data
├── best_model_half/                # Half-trained model files
│   ├── tf_model.h5
│   ├── config.json
│   ├── vocab.txt
│   └── elect_draft8.py
├── best_model_all_best/            # Best performing model
│   ├── tf_model.h5
│   ├── config.json
│   ├── saved_model.pb
│   └── variables/
└── best_model_all_scores.docx      # Model performance documentation
```

## 🛠️ Key Features

### 1. **Real-time Data Collection**
- Fetches top posts from any subreddit
- Configurable time filters (day, week, month)
- Extracts comments and metadata
- Handles Reddit API rate limits

### 2. **Advanced Text Preprocessing**
- Contraction expansion ("don't" → "do not")
- URL and special character removal
- Text normalization and lemmatization
- Stop word removal

### 3. **ELECTRA-based Sentiment Analysis**
- Fine-tuned ELECTRA model for 3-class classification
- Labels: Positive, Neutral, Negative
- Real-time prediction on new text
- Confidence scores for predictions

### 4. **Interactive Web Interface**
- Streamlit-based dashboard
- Real-time parameter adjustment
- Interactive visualizations
- Data export capabilities

### 5. **Comprehensive Analytics**
- Sentiment trends over time
- Word frequency analysis
- Comment length distribution
- Named Entity Recognition
- Period-based aggregation

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install streamlit
pip install praw
pip install spacy
pip install transformers
pip install torch
pip install pandas
pip install matplotlib
pip install seaborn
pip install nltk
pip install contractions
```

### Additional Setup
```bash
# Download spaCy English model
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('wordnet')"
```

### Reddit API Setup
1. Create a Reddit app at https://www.reddit.com/prefs/apps
2. Get your client_id and client_secret
3. Update the credentials in `Main.py` (lines 124-126)

## 🚀 Usage

### Running the Main Application
```bash
streamlit run Main.py
```

### Key Parameters
- **Subreddit Name**: Target subreddit (default: 'wallstreetbets')
- **Time Filter**: 'day', 'week', or 'month'
- **Number of Posts**: Posts to analyze (default: 10)
- **Number of Comments**: Comments per post (default: 5)
- **Analysis Period**: 'D' (daily), 'W' (weekly), or 'M' (monthly)

### Features Available
1. **Sentiment Analysis**: Real-time classification of Reddit comments
2. **Time Series Visualization**: Sentiment trends over selected periods
3. **Text Analytics**: Word frequency and comment length analysis
4. **Entity Recognition**: Extract people, organizations, locations, etc.
5. **Data Export**: Download processed data and visualizations

## 📊 Model Performance

The project includes two model variants:
- **best_model_half**: Partially trained model (418MB)
- **best_model_all_best**: Fully trained best-performing model (52MB)

Model performance metrics are documented in `best_model_all_scores.docx`.

## 🔍 Technical Details

### Model Architecture
- **Base Model**: ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
- **Task**: 3-class sentiment classification
- **Input**: Preprocessed Reddit text
- **Output**: Sentiment labels with confidence scores

### Data Processing Pipeline
1. **Scraping**: PRAW → Reddit API → Raw data
2. **Preprocessing**: Text cleaning → Lemmatization → Tokenization
3. **Classification**: ELECTRA model → Sentiment prediction
4. **Visualization**: Matplotlib/Seaborn → Interactive plots

### Performance Optimizations
- Pre-loaded NLP models for faster processing
- Batch processing for multiple texts
- Efficient data structures for large datasets
- Caching for repeated operations

## 📈 Use Cases

1. **Market Sentiment Analysis**: Track sentiment around stocks/crypto
2. **Brand Monitoring**: Analyze public opinion about products/services
3. **Social Media Research**: Study trends and public discourse
4. **Academic Research**: Sentiment analysis for research papers
5. **Content Moderation**: Automated sentiment-based filtering

## 🤝 Contributing

This project was developed as part of a Big Data Analytics course. Key components:
- **Data Collection**: Reddit API integration
- **Model Training**: ELECTRA fine-tuning
- **Visualization**: Interactive dashboards
- **Deployment**: Streamlit web application

## 📝 Notes

- The model requires significant computational resources for training
- Reddit API has rate limits that may affect data collection
- Model performance may vary across different subreddits
- Regular retraining recommended for optimal performance

## 🔗 Dependencies

- **Streamlit**: Web application framework
- **PRAW**: Reddit API wrapper
- **Transformers**: Hugging Face transformer models
- **spaCy**: Natural language processing
- **PyTorch**: Deep learning framework
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization

---

*This project demonstrates the application of state-of-the-art transformer models for real-time sentiment analysis on social media data.*
