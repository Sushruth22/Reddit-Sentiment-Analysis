#%%
import streamlit as st
import praw, re, spacy, torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from datetime import datetime
from nltk.stem import WordNetLemmatizer
import contractions
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from collections import Counter
import seaborn as sns

# Pre-loading NLP resources
nlp = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()

# Load trained model
model_path = 'D:/Big-Data/Electra/best_model_half'
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path, from_tf=True, num_labels=3)

def get_reddit_data(subreddit_name, time_filter, num_posts, num_comments, reddit):
    subreddit = reddit.subreddit(subreddit_name)
    reddit_data = []
    for post in subreddit.top(time_filter=time_filter, limit=num_posts):
        comments = [comment for comment in post.comments[:num_comments+1] if not comment.stickied]
        reddit_data.extend({
            'post_title': post.title,
            'post_text': post.selftext,
            'post_created': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'comment_text': comment.body,
            'comment_created': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        } for comment in comments)
    return reddit_data

def preprocess_text(texts):
    fixed_texts = [contractions.fix(text) for text in texts]
    cleaned_texts = [re.sub(r'http\S+|[^a-zA-Z0-9]', ' ', text).lower() for text in fixed_texts]
    cleaned_texts = [re.sub(r' +', ' ', text) for text in cleaned_texts]
    docs = list(nlp.pipe(cleaned_texts))
    return [' '.join([lemmatizer.lemmatize(token.text) for token in doc]) for doc in docs]

def get_period(df, period):
    df['comment_created'] = pd.to_datetime(df['comment_created'])
    df['period_date'] = df['comment_created'].dt.to_period(period).dt.to_timestamp()
    df['period_int'] = df['period_date'].rank(method='dense').astype(int)
    return df

def apply_model(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = softmax(logits, dim=1)
    class_predictions = torch.argmax(predictions, dim=1)
    labels = {0: 'neutral', 1: 'positive', 2: 'negative'}
    class_prediction_labels = [labels[pred.item()] for pred in class_predictions]
    return class_predictions, class_prediction_labels

def plot_sentiment_over_time(df):
    fig, ax = plt.subplots()
    for sentiment in ['positive', 'neutral', 'negative']:
        df_sentiment = df[df['sentiment_labels'] == sentiment]
        df_sentiment.groupby('period_int').size().plot(ax=ax, label=sentiment)
    plt.title('Sentiment Analysis Over Time')
    plt.xlabel('Period Beginning')
    plt.ylabel('Number of Comments')
    plt.legend()
    st.pyplot(fig)

def table_sentiment_counts_per_period(df):
    df_counts = df.groupby(['period_int', 'sentiment_labels']).size().unstack(fill_value=0)
    df_counts['total'] = df_counts.sum(axis=1)
    return df_counts


def ner_on_text(text):
    # get named entities
    doc = nlp(text)
    entity_types=['PERSON', 'ORG', 'GPE', 'NORP', 'PRODUCT', 'EVENT', 'WORK_OF_ART'] # person, organization, geopolitical entity; what else? 
    #entities = [(ent.text) for ent in doc.ents]
    entities = [(ent.text) for ent in doc.ents if ent.label_ in entity_types]
    return entities

def plot_word_frequency(texts, n_words=20):
    # remove stopwords
    stop_words = nlp.Defaults.stop_words
    texts = [' '.join([word for word in text.split() if word not in stop_words]) for text in texts] # this is a list comprehension to remove stopwords from each text in the list of texts; it returns a list of texts

    # Combine all texts into a single list and count
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    
    # Select the most common words and convert to df
    most_common_words = word_counts.most_common(n_words)
    df_words = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Word', data=df_words, palette='viridis')
    plt.title(f'Top {n_words} Most Common Words')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    st.pyplot(plt.gcf())

def plot_comment_length_distribution(texts):
    # Calculate comment length
    comment_lengths = [len(text.split()) for text in texts]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(comment_lengths, bins=30, kde=True, color='skyblue')
    plt.title('Comment Length Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())


def main():
    st.title('Reddit Sentiment Analysis')

    # Reddit API credentials setup
    reddit = praw.Reddit(client_id='9eqVQ87jVyzQE8dXhMLRRQ',
                         client_secret='9KXzo5bnpQ26c51BRxyS2QyY6k641Q',
                         user_agent='gwu_bigdata6450')

    # Streamlit widgets to input parameters
    subreddit_name = st.sidebar.text_input('Subreddit Name', 'wallstreetbets')
    time_filter = st.sidebar.selectbox('Time Filter', ['day', 'week', 'month'])
    num_posts = st.sidebar.number_input('Number of Posts', min_value=1, value=10)
    num_comments = st.sidebar.number_input('Number of Comments', min_value=1, value=5)
    period = st.sidebar.selectbox('Analysis Period', ['D', 'W', 'M'])
    show_text_analysis = st.sidebar.checkbox('Show Text Analysis', False)
    
    # Run analysis
    if st.sidebar.button('Run Analysis'):
        # Fetch data and preprocess it
        df_raw = pd.DataFrame(get_reddit_data(subreddit_name, time_filter, num_posts, num_comments+1, reddit))
        df_raw['clean_text'] = preprocess_text(df_raw['comment_text'].tolist())
        
        # Apply model
        _, class_prediction_labels = apply_model(df_raw['clean_text'].tolist(), model, tokenizer)
        df_raw['sentiment_labels'] = class_prediction_labels
        
        # Period processing
        df_clean = get_period(df_raw, period=period)

        # Named Entity Recognition
        df_clean['named_entities'] = df_clean['post_title'].apply(ner_on_text) + df_clean['post_text'].apply(ner_on_text) + df_clean['comment_text'].apply(ner_on_text) # NER on comment text + post text
        
        # Display results
        st.write('Processed Data with Sentiment Labels:')
        st.dataframe(df_clean[['comment_text', 'sentiment_labels', 'comment_created', 'period_int', 'post_title', 'named_entities']])
        
        # Display sentiment trends and counts side by side
        st.write('Sentiment Analysis:')
        col1, col2 = st.columns(2)
        with col1:
            plot_sentiment_over_time(df_clean)
        with col2:
            st.write('Sentiment Counts per Period:')
            st.dataframe(table_sentiment_counts_per_period(df_clean))

        # Text Analysis if the option is selected on query
        if show_text_analysis:
            st.write('Text Analysis:')
            
            # Show Word Frequency Distribution and Token Length Distribution side by side
            col3, col4 = st.columns(2)
            with col3:
                plot_word_frequency(df_clean['clean_text'].tolist())
            with col4:
                plot_comment_length_distribution(df_clean['clean_text'].tolist())

        # Analysis complete
        st.write('End')

if __name__ == "__main__":
    st.set_page_config(page_title='Reddit Sentiment Analysis', layout='wide')
    main()

# run the app with below line in terminal
# streamlit run FinalFiles/Main.py

