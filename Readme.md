# Reddit Sentiment Analysis with ELECTRA

A comprehensive sentiment analysis application that uses the ELECTRA transformer model to analyze Reddit comments and posts in real-time. The project includes data scraping, preprocessing, model training, and interactive visualization components.

##  Project Overview

This project implements a complete pipeline for:
- **Real-time Reddit data scraping** using PRAW (Python Reddit API Wrapper)
- **Text preprocessing** with spaCy and NLTK
- **Sentiment classification** using a fine-tuned ELECTRA model
- **Interactive web interface** built with Streamlit
- **Data visualization** with matplotlib and seaborn
- **Named Entity Recognition** for extracting important entities

##  Project Structure

``nElectra/
 Main.py                          # Main Streamlit application
 RedditScrape_draft4.py          # Reddit data scraping utilities
 elect_draft4.py                 # ELECTRA model training script
 df.csv                          # Main dataset (13MB)
 df_all.csv                      # Extended dataset (13MB)
 cleaned_reddit_data.csv         # Processed Reddit data
 raw_reddit_data.csv             # Raw Reddit data
 best_model_half/                # Half-trained model files
 best_model_all_best/            # Best performing model
 best_model_all_scores.docx      # Model performance documentation
``
