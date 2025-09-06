#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFTrainer, TFTrainingArguments
from transformers import ElectraTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.metrics import Precision, Recall
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall


def preprocess_text(text, nlp, lemmatizer):
    # Clean text
    #text = re.sub(r'http\S+|[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text).lower()
    text = re.sub(' +', ' ', text) # 

    # lemma; stopwords removed reduced accuracy
    doc = nlp(text)
    cleaned_text = ' '.join([lemmatizer.lemmatize(token.text) for token in doc])# if token.text not in stop_words])

    return cleaned_text

def train_model(df, model_name='google/electra-small-discriminator', epochs=15, batch_size=128):
    # Ensure tokenizer is defined with the given model name
    tokenizer = ElectraTokenizer.from_pretrained(model_name)

    # label mapping to ensure model compatibility; cannot take -1 as a label
    label_mapping = {-1: 2, 0: 0, 1: 1} # -1 (2) is negative, 0 (0) is neutral, 1 (1) is positive
    df['sentiment'] = df['sentiment'].map(label_mapping)

    # Split data into training, validation, and test sets
    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # Tokenize data function (as previously defined)
    def tokenize_data(texts, tokenizer):
        input_ids, attention_masks = [], []
        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                text,                        # Text to encode.
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                max_length=128,             # Pad & truncate all sentences.
                #pad_to_max_length=True,      # Pad sentence to max length.
                padding='max_length',
                return_attention_mask=True,  # Return attention mask.
                return_tensors='tf',         # Return TensorFlow tensors.
                truncation=True              # Explicitly truncate examples to max length.
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = tf.concat(input_ids, 0)
        attention_masks = tf.concat(attention_masks, 0)
        return input_ids, attention_masks

    # Tokenize all sets
    input_ids_train, attention_masks_train = tokenize_data(X_train, tokenizer)
    input_ids_val, attention_masks_val = tokenize_data(X_val, tokenizer)
    input_ids_test, attention_masks_test = tokenize_data(X_test, tokenizer)

    # Convert labels to numpy for training, validation, and testing
    #labels_train = np.array(y_train)
    #labels_val = np.array(y_val)
    #labels_test = np.array(y_test)

    # Convert labels to one-hot encoding
    labels_train = to_categorical(y_train, num_classes=3)
    labels_val = to_categorical(y_val, num_classes=3)
    labels_test = to_categorical(y_test, num_classes=3)

    # Load and compile the Electra model for sequence classification
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(y)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy', Precision(), Recall()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Initialize ModelCheckpoint callback to save best model
    model_checkpoint_callback = ModelCheckpoint(
        filepath='c:/Users/schil/OneDrive/Desktop/School/6450_BigData/Project/Model_electra/best_model', # Specify your path and file name here
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1, 
        #save_format='tf'
        )

    # Train the model
    model.fit([input_ids_train, attention_masks_train], labels_train, 
              validation_data=([input_ids_val, attention_masks_val], labels_val),
              epochs=epochs, batch_size=batch_size, callbacks=[model_checkpoint_callback])

    # Evaluate the model (returns test loss, accuracy, precision, recall)
    eval_result = model.evaluate([input_ids_test, attention_masks_test], labels_test)

    # Return the trained model and its evaluation result on the test set
    return model, eval_result


# load data
print('loading data...')
df = pd.read_csv('df_all.csv')
#df = df.sample(frac=0.5)
print('loaded')

# clean data - 50% 
print('cleaning data...')
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: preprocess_text(x, nlp, lemmatizer))
print('cleaned')

# train model
print('training model...')
model, eval_result = train_model(df, model_name='google/electra-small-discriminator', epochs=5, batch_size=64)
print(eval_result)
# evaluation results
precision = eval_result[2]
recall = eval_result[3]
f1 = 2 * (precision * recall) / (precision + recall)
print(f"\nEvaluation Result on Test Data:\n- Loss: {eval_result[0]}\n- Accuracy: {eval_result[1]}\n- Precision: {precision}\n- Recall: {recall}\n- F1 Score: {f1}")

# get model summary
print()
print(model.summary())
print()

