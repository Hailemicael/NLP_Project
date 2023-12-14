# -*- coding: utf-8 -*-
"""NLP_Assignment_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sxiA6kK6YF7lqxEwWAjD67IHnVaQvcyZ
"""

!pip install wikipedia-api

# Import libraries for our project
import pandas as pd
import wikipediaapi
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('NLP_Project(hailelulseged281913@gmail.com)', 'en')

# List of medical and non-medical keywords
medical_keywords = ["Medicine", "Cardiology", "Surgery", "Health", "Pharmacy", "Immunology",
                    "Pathology", "Pediatrics", "Oncology", "Neurology", "Dentistry", "vascular",
                    "orthopedic", "dermatology", "endocrinology", "gastroenterology", "pulmonary", "neurosurgery",
                    "ophthalmology", "radiology", "anesthesiology", "genetics", "oncologist", "hematology",
                    "immunotherapy", "pediatrician", "psychiatry", "dentist", "Anatomy",
                    "Physiology", "Biochemistry", "balanced diet", "Ailment", "Affliction", "Illness", "Sickness",
                    "Hereditary", "Infectious", "Pandemic", "nurse", "Doctor", "Alzheimer", "virus", "surgery"]

non_medical_keywords = ["Art", "Literature", "Philosophy", "Science", "Technology", "Space",
                        "Environment", "Food", "Cuisine", "Recipes", "Cooking", "History", "Ancient_Civilizations",
                        "Archaeology", "painting", "sculpture", "literary", "fiction", "poetry", "philosopher",
                        "culinary", "gastronomy", "recipe", "culinary", "history", "historical", "architectural",
                        "archaeological", "civilization", "culture", "Engineering", "Astronomy", "Cosmology", "country", "Industry", "ocean", "charger", "battery", "music", "dance", "painting", "sculpture", "artistic", "novel"]

# Function to fetch content from Wikipedia using wikipediaapi
def fetch_content(title, wiki_wiki):
    # Fetches content from Wikipedia for a given title.
    page_py = wiki_wiki.page(title)
    content = page_py.text

    return content

# Fetch content for medical keywords
medical_content_list = [fetch_content(keyword, wiki_wiki) for keyword in medical_keywords]
print("Fetched content for medical keywords:")
print(medical_content_list)
# Fetch content for non-medical keywords
non_medical_content_list = [fetch_content(keyword, wiki_wiki) for keyword in non_medical_keywords]
print("Fetched content for non-medical keywords:")
print(non_medical_content_list)

# Function to fetch content from Wikipedia using wikipediaapi
def fetch_content(title, wiki_wiki):
    # Fetches content from Wikipedia for a given title.
    page_py = wiki_wiki.page(title)
    content = page_py.text

    # Clean the content
    cleaned_content = clean_text(content)

    return cleaned_content

# Function to clean text (remove HTML tags, references, etc.)
def clean_text(text):
    # Remove HTML tags and comments
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")

    # Remove special characters and non-alphabetic characters
    clean_text = re.sub(r"[^a-zA-Z\s]", "", clean_text)

    # Tokenize the text
    tokens = word_tokenize(clean_text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove extra whitespaces
    clean_text = " ".join(tokens)

    return clean_text
# Clean medical content
cleaned_medical_content_list = [clean_text(content) for content in medical_content_list]
print("Text cleaning applied to medical content:")
print(cleaned_medical_content_list)
# Clean non-medical content
cleaned_non_medical_content_list = [clean_text(content) for content in non_medical_content_list]
print("Text cleaning applied to non-medical content:")
print(cleaned_non_medical_content_list)

# Create a DataFrame with the fetched and cleaned data
medical_data = {"text": cleaned_medical_content_list, "label": ["medical"] * len(cleaned_medical_content_list)}
non_medical_data = {"text": cleaned_non_medical_content_list, "label": ["non-medical"] * len(cleaned_non_medical_content_list)}

df_medical = pd.DataFrame(medical_data)
df_non_medical = pd.DataFrame(non_medical_data)
# Concatenate the dataframes and shuffle rows
df = pd.concat([df_medical, df_non_medical], ignore_index=True).sample(frac=1)

# Check the updated dataset
print("Updated Dataset:")
print(df)
df.to_csv('medical_non_medical_dataset.csv', index=False)
print("Dataset saved as 'medical_non_medical_dataset'")

"""# **Model Training Part by Naive_bayes**"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
df = pd.read_csv('medical_non_medical_dataset.csv')

# Check class distribution
print("Original Data Distribution:")
print(df['label'].value_counts())

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=100)

# Handle missing values in X_test
X_test = X_test.fillna('')  # Replace NaN with an empty string or any other placeholder

# Handle missing values in X_train
X_train = X_train.fillna('')  # Replace NaN with an empty string or any other placeholder

# Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Resampling using SMOTE
sampler = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_tfidf, y_train)

# Model Selection and Training (Naive Bayes)
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation (Naive Bayes)
nb_predictions = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"\nNaive Bayes Accuracy: {nb_accuracy}")
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))

# Example of making predictions on Naive Bayes
new_data_extended = [
    "heart disease",
    "expression varies across cultures.",
    "New year in Ethiopia is good.",
    "my health is not good",
    "doctors support patient",
]

new_data_extended_tfidf = tfidf_vectorizer.transform(new_data_extended)
new_predictions_extended = nb_model.predict(new_data_extended_tfidf)

print("\nNaive Bayes Predictions :")
for text, prediction in zip(new_data_extended, new_predictions_extended):
    print(f"{text} - Predicted: {prediction}")

"""# **Model Training Part by Logistic Regression**"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the cleaned dataset
df = pd.read_csv('medical_non_medical_dataset.csv')

# Check class distribution
print("Original Data Distribution:")
print(df['label'].value_counts())

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=100)

# Handle missing values in X_test
X_test = X_test.fillna('')  # Replace NaN with an empty string or any other placeholder

# Handle missing values in X_train
X_train = X_train.fillna('')  # Replace NaN with an empty string or any other placeholder

# Feature Extraction (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Resampling using SMOTE
sampler = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_tfidf, y_train)

# Model Selection and Training (Logistic Regression)
model = LogisticRegression(random_state=100)
model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, predictions))

# Example of making predictions on more new data using Logistic Regression
new_data_extended = [
    "heart disease",
    "expression varies across cultures.",
    "New year in Ethiopia is good."
]

new_data_extended_tfidf = tfidf_vectorizer.transform(new_data_extended)
new_predictions_extended = model.predict(new_data_extended_tfidf)

print("\nLogistic Regression Predictions on more new data:")
for text, prediction in zip(new_data_extended, new_predictions_extended):
    print(f"{text} - Predicted: {prediction}")