#%%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import Trainer, TFTrainingArguments
from transformers import ElectraTokenizer, TFAutoModelForSequenceClassification
import keras
from keras.metrics import Precision, Recall
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk.stem import WordNetLemmatizer
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.metrics import Precision, Recall

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

def remove_outlier_lengths(df):
    # Remove outliers
    #max_len = df['text'].apply(lambda x: len(x.split())).max()
    #print(f'Max Length of Text: {max_len}') 

    # Split lengths of text into bins
    df['text_len'] = df['text'].apply(lambda x: len(x.split()))
    #print(df['text_len'].value_counts(bins=10, sort=False))

    # Remove outliers
    df = df[df['text_len'] <= 1000]
    df = df[df['text_len'] >= 2]
    return df

def train_model(df, model_name='google/electra-base-discriminator', epochs=3, batch_size=64):
    # Ensure tokenizer is defined with the given model name
    tokenizer = ElectraTokenizer.from_pretrained(model_name)

    # label mapping to ensure model compatibility; cannot take -1 as a label
    label_mapping = {-1: 2, 0: 0, 1: 1} # -1 (2) is negative, 0 (0) is neutral, 1 (1) is positive
    df['sentiment'] = df['sentiment'].map(label_mapping)

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # Tokenize data function (as previously defined)
    def tokenize_data(texts, tokenizer):
        input_ids, attention_masks = [], []
        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                text,                        # Text to encode.
                add_special_tokens=True,     # Add '[CLS]' and '[SEP]'
                max_length=300, #150            # Pad & truncate all sentences.
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

    # Convert labels to one-hot encoding
    labels_train = to_categorical(y_train, num_classes=3)
    labels_val = to_categorical(y_val, num_classes=3)
    labels_test = to_categorical(y_test, num_classes=3)

    # Load and compile the Electra model for sequence classification
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy', Precision(), Recall()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    history = model.fit([input_ids_train, attention_masks_train], labels_train, 
                        validation_data=([input_ids_val, attention_masks_val], labels_val),
                        epochs=epochs, 
                        batch_size=batch_size, 
                        #callbacks=[model_checkpoint_callback]
                        )
  
    # Save model and tokenizer conditionally to get best one
    best_model_path = 'C:/Users/schil/OneDrive/Desktop/School/6450_BigData/electra_best/best_model5'
    val_accuracy = history.history['val_accuracy']

    if max(val_accuracy) == val_accuracy[-1]:  # Check if the last epoch had the best validation accuracy
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)

    # Evaluate the model (returns test loss, accuracy, precision, recall)
    eval_result = model.evaluate([input_ids_test, attention_masks_test], labels_test)

    # Return the trained model and its evaluation result on the test set
    return model, eval_result


# load data
print('loading data...')
df = pd.read_csv('C:/Users/schil/OneDrive/Desktop/School/6450_BigData/df_all.csv')
df = df.sample(frac=0.5)
print('loaded')

# clean data - 50% 
print('cleaning data...')
nlp = spacy.load("en_core_web_lg") # download enc_core_web_md with this command in the terminal: python -m spacy download en_core_web_md
lemmatizer = WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: preprocess_text(x, nlp, lemmatizer))
df = remove_outlier_lengths(df)
print('cleaned')

# train model
print('training model...')
model, eval_result = train_model(df, model_name='google/electra-base-discriminator', epochs=3, batch_size=16)
print(eval_result)
# evaluation results
precision = eval_result[2]
recall = eval_result[3]
f1 = 2 * (precision * recall) / (precision + recall)
print(f"\nEvaluation Result on Test Data:\n- Loss: {eval_result[0]}\n- Accuracy: {eval_result[1]}\n- Precision: {precision}\n- Recall: {recall}\n- F1 Score: {f1}")

# get model summary
print()
#print(model.summary())
print('=='*50)


#%%
# load model
model = TFAutoModelForSequenceClassification.from_pretrained('C:/Users/schil/OneDrive/Desktop/School/6450_BigData/electra_best/best_model4/')
tokenizer = ElectraTokenizer.from_pretrained('C:/Users/schil/OneDrive/Desktop/School/6450_BigData/electra_best/best_model4')

#%%
# print curdir
import os
print(os.getcwd())



#%%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import keras
print("Keras version:", keras.__version__)

import transformers
print("Transformers version:", transformers.__version__)



#%%
# download en_core_web_lg
import spacy
spacy.cli.download("en_core_web_lg")

#%%

# download wordnet
import nltk
nltk.download('wordnet')




















































#%%
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import TFElectraForSequenceClassification, ElectraTokenizer, ElectraConfig
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import logging
import re
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

class ElectraClassifier:

    def __init__(self, label_names, model_name='google/electra-small-discriminator'):
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_names = label_names
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

    def load_data_from_jsonl(self, filename):
        texts, labels = [], []
        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    data = json.loads(line)
                    texts.append(data['text'])
                    labels.append(data['label'])
                except json.JSONDecodeError as e:
                    print(f"Error in line {line_number}: {e}")
                    break

        input_ids, attention_masks = [], []
        for text in texts:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='tf',
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = tf.concat(input_ids, 0)
        attention_masks = tf.concat(attention_masks, 0)
        labels = np.array(labels)
        return input_ids, attention_masks, labels, texts


    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        text = re.sub(' +', ' ', text)

        # Lemmatize
        doc = self.nlp(text)
        text = ' '.join([self.lemmatizer.lemmatize(token.text) for token in doc])

        return text
    def create_model(self, num_labels=6):
        config = ElectraConfig.from_pretrained('google/electra-small-discriminator', num_labels=num_labels)
        self.model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', config=config)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train_model(self, train_data, validation_data, epochs=15, batch_size=128):
        model_path = os.path.join(os.getcwd(), 'best_model_electra')
        logging.info(f"Model and tokenizer will be saved to: {model_path}")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3)
        ]

        try:
            history = self.model.fit(
                train_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )

            # Save the model and the tokenizer
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

        return history

    def evaluate_model(self, validation_data, texts_validation, label_names):
        input_ids_test, attention_masks_test, labels_test = validation_data.values()
        texts_test_series = pd.Series(texts_validation, name='Text')

        y_pred_logits = self.model.predict({'input_ids': input_ids_test, 'attention_mask': attention_masks_test}).logits
        y_pred_scores = tf.nn.softmax(y_pred_logits, axis=1).numpy()
        y_pred_labels = tf.argmax(y_pred_scores, axis=1).numpy()

        scores_df = pd.DataFrame(y_pred_scores, columns=label_names)
        final_df = pd.concat([texts_test_series, scores_df], axis=1)
        final_df['Overall_Score'] = final_df[label_names].max(axis=1)

        report = classification_report(labels_test, y_pred_labels, target_names=label_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        return final_df, report_df

    # model_path is the directory to tf model
    def load_model(self, model_path):
        return TFElectraForSequenceClassification.from_pretrained(model_path)

    def infer(self, model, text):

        input_ids, attention_masks = [], []
        for text in text:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='tf',
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = tf.concat(input_ids, 0)
        attention_masks = tf.concat(attention_masks, 0)

        predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_masks})
        predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

        predicted_labels = [self.label_names[label] for label in predicted_labels]

        return predicted_labels

def main():

    label_names = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    classifier = ElectraClassifier(label_names)
    input_ids, attention_masks, labels, texts = classifier.load_data_from_jsonl('hug_data.jsonl', )
    classifier.create_model(num_labels=6)

    # Convert tensors to NumPy arrays
    if isinstance(input_ids, tf.Tensor):
        input_ids = input_ids.numpy()
    if isinstance(attention_masks, tf.Tensor):
        attention_masks = attention_masks.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=2018)
    train_masks, validation_masks, texts_train, texts_validation = train_test_split(attention_masks, texts, test_size=0.2, random_state=2018)

    train_inputs = tf.convert_to_tensor(train_inputs)
    validation_inputs = tf.convert_to_tensor(validation_inputs)
    train_masks = tf.convert_to_tensor(train_masks)
    validation_masks = tf.convert_to_tensor(validation_masks)
    train_labels = tf.convert_to_tensor(train_labels)
    validation_labels = tf.convert_to_tensor(validation_labels)

    train_data = {'input_ids': train_inputs, 'attention_mask': train_masks, 'labels': train_labels}
    validation_data = {'input_ids': validation_inputs, 'attention_mask': validation_masks, 'labels': validation_labels}

    train_history = classifier.train_model(train_data, validation_data)
    final_df, report_df = classifier.evaluate_model(validation_data, texts_validation, label_names)

    print("Evaluation Scores:")
    print(final_df.head())

    print("\nClassification Report:")
    print(report_df)


def from_pretrained(model_path):
    print("here")
    label_names = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    classifier = ElectraClassifier(label_names)

    # Load the model and the tokenizer
    classifier.model = TFElectraForSequenceClassification.from_pretrained(model_path)
    classifier.tokenizer = ElectraTokenizer.from_pretrained(model_path)

    sentiment = classifier.infer(classifier.model, ['Please figure out the sentiment for this text. Scared if it actually works'])
    print(sentiment)

if __name__ == "__main__":
    from_pretrained('best_model_electra')

