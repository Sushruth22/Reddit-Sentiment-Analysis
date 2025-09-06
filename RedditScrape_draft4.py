#%%
import praw, re, spacy, torch
from torch.nn.functional import softmax
from datetime import datetime
import pandas as pd
from nltk.stem import WordNetLemmatizer
import contractions
from transformers import ElectraForPreTraining, ElectraTokenizer, ElectraForSequenceClassification

def get_reddit_data(subreddit_name, time_filter, num_posts, num_comments, reddit):
    '''
    Function to fetch data from a specified subreddit using the Reddit API.
    It will fetch the top num_comments (first layer), from the top num_posts of the 
    subreddit, over the specified time_filter.
        - Example: top 3 comments from each of the top 20 posts over the previous week
            from r/wallstreetbets
    '''
    subreddit = reddit.subreddit(subreddit_name)
    reddit_data = []
    for post in subreddit.top(time_filter=time_filter, limit=num_posts):
        comments = [comment for comment in post.comments[:num_comments+1] if not comment.stickied]
        reddit_data.extend({
            'comment_text': comment.body,
            'comment_created': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
        } for comment in comments)
    return reddit_data


def preprocess_text(texts):
    fixed_texts = [contractions.fix(text) for text in texts]
    cleaned_texts = [re.sub(r'http\S+|[^a-zA-Z0-9.;:!?\"-]', ' ', text).lower() for text in fixed_texts]
    cleaned_texts = [re.sub(r' +', ' ', text) for text in cleaned_texts]
    docs = list(nlp.pipe(cleaned_texts))
    return [' '.join([lemmatizer.lemmatize(token.text) for token in doc]) for doc in docs]


def get_period(df, period):
    df['comment_created'] = pd.to_datetime(df['comment_created'])
    df['period_date'] = df['comment_created'].dt.to_period(period).dt.to_timestamp()
    df['period_int'] = df['period_date'].rank(method='dense').astype(int)
    return df


def apply_model(text, model, tokenizer):
    # Tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class probabilities
    logits = outputs.logits
    predictions = softmax(logits, dim=1)

    # Get predicted class labels
    class_predictions = torch.argmax(predictions, dim=1)

    # Map numerical class predictions to labels
    labels = {0: 'neutral', 1: 'positive', 2: 'negative'}
    class_prediction_labels = [labels[pred.item()] for pred in class_predictions]

    return class_predictions, class_prediction_labels


def main():
    '''
    The main function scrapes the reddit data live, preprocesses it, 
        and applies the model. Then comments are grouped according to their 
        time stampts and assigned a "period number" to enable analysis of
        the changes in sentiment of time, before being saved into csv files. 
    
    Parameters:
     - subreddit_name: The name of the subreddit from which data is to be fetched.
     - time_filter: A filter specifying the time period for the posts (such as 'day', 'week', 'month', etc.).
     - num_posts: The number of top posts to retrieve from the subreddit.
     - num_comments: The maximum number of comments to retrieve from each post. 
            Must +1 to account for the post itself. 
     - reddit: An instance of a Reddit client, typically created using a library like praw, which is used to interact with the Reddit API.
     - period: The time period to group the comments by (such as 'D' for day, 'W' for week, etc.).
     - model: A trained model that can be used to classify the sentiment of text data.
        o tokenizer: A tokenizer that can be used to preprocess text data before passing it to the model.

    Output:
        - raw_reddit_data.csv: a csv file containing the raw scraped data.
        - cleaned_reddit_data.csv: a csv file containing the cleaned data, with sentiment analysis labels.
    '''
    # initialize Reddit API
    reddit = praw.Reddit(client_id='9eqVQ87jVyzQE8dXhMLRRQ', 
                         client_secret='9KXzo5bnpQ26c51BRxyS2QyY6k641Q', 
                         user_agent='gwu_bigdata6450')

    # parameters + scrape reddit data
    # ex: scrape the top 3 comments from each of the top 5 posts in 
        # r/wallstreetbets over the past month
    subreddit_name = 'gwu'
    time_filter = 'month'
    num_posts = 5
    num_comments = 3
    df_raw = pd.DataFrame(get_reddit_data(subreddit_name, time_filter, num_posts, num_comments+1, reddit))

    # preprocess scraped comments
    df_raw['clean_text'] = preprocess_text(df_raw['comment_text'].tolist())

    # apply model to preprocessed text
    class_predictions, class_prediction_labels = apply_model(df_raw['clean_text'].tolist(), model, tokenizer)
    df_raw['class_predictions'] = class_predictions
    df_raw['sentiment_labels'] = class_prediction_labels

    # period processing
    period = 'W'
    df_clean = get_period(df_raw, period=period)
    
    # Save results into csv files
    df_raw.to_csv('raw_reddit_data.csv', index=False)
    df_clean.to_csv('cleaned_reddit_data.csv', index=False)

    print(df_clean.shape)



# Pre-loading NLP resources
nlp = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()

# Load trained model
tokenizer = ElectraTokenizer.from_pretrained('D:/Big-Data/Electra/best_model_half')
model = ElectraForSequenceClassification.from_pretrained('D:/Big-Data/Electra/best_model_half', from_tf=True, num_labels=3)

if __name__ == "__main__":
    main()

