# NLP_Project/Text Classification of Medical and Non-Medical Content Using Wikipedia/
Natural Language Processing Assignment on Text Classification 
## Overview
This project aims to categorize texts, specifically in the English language, into two distinct categories: medical and non-medical. The implementation employs a range of natural language processing (NLP) techniques and ML Algorithms.

## Dataset Preparation
### Fetching and Cleaning Wikipedia Content
The dataset is created by fetching content from Wikipedia for a predefined set of medical and non-medical keywords. The Wikipedia API, accessed through the `wikipediaapi` Python library, is used to obtain relevant articles. The content is then cleaned using `BeautifulSoup` to remove HTML tags, comments, and other artifacts. Additionally, text is tokenized, stop words are removed, and lemmatization is performed using NLTK functions.

#### Medical Keywords
- A list of medical keywords, including terms related to various medical specialties, diseases, and healthcare.

#### Non-Medical Keywords
- A list of non-medical keywords, covering various topics such as art, literature, philosophy, and more.

### Creating the Dataset
The cleaned content is organized into a pandas DataFrame with columns "text" and "label" where "text" contains the cleaned content, and "label" indicates whether the text is medical or non-medical. The dataset is then shuffled for better randomness.

```python
# Example Code for Creating the Dataset
df = pd.concat([df_medical, df_non_medical], ignore_index=True).sample(frac=1)
df.to_csv('medical_non_medical_dataset.csv', index=False)
```
## Model Training
I have used both Naive Bayes and logistic regression.
### Naive Bayes Approach
The dataset is loaded, and missing values are handled appropriately. Text data is converted into TF-IDF vectors using the TfidfVectorizer from scikit-learn. To address the class imbalance, the dataset is resampled using SMOTE (Synthetic Minority Over-sampling Technique). The Naive Bayes model (MultinomialNB) is then trained using the resampled data.

```python
# Example Code for Naive Bayes Model Training
nb_model = MultinomialNB()
nb_model.fit(X_train_resampled, y_train_resampled)
```
### Logistic Regression Approach
Similar to the Naive Bayes approach, Logistic Regression is employed for classification. The TF-IDF vectors are used as features for training. Resampling is applied using SMOTE, and the Logistic Regression model is trained.

```python
# Example Code for Logistic Regression Model Training
model = LogisticRegression(random_state=100)
model.fit(X_train_resampled, y_train_resampled)
```
## Model Evaluation and Predictions
The trained models are evaluated using test data, and accuracy scores and classification reports are generated.

### Naive Bayes Model Evaluation
```python
# Example Code for Naive Bayes Model Evaluation
nb_predictions = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"\nNaive Bayes Accuracy: {nb_accuracy}")
print("Naive Bayes Classification Report:\n", classification_report(y_test, nb_predictions))
```
### Logistic Regression Model Evaluation
```python
# Example Code for Logistic Regression Model Evaluation
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, predictions))
```
## Predictions on New Data
Predictions are made on new data using both Naive Bayes and Logistic Regression models.

```python
# Example Code for Predictions on New Data
new_data_extended_tfidf = tfidf_vectorizer.transform(new_data_extended)
new_predictions_extended = nb_model.predict(new_data_extended_tfidf)

print("\nNaive Bayes Predictions on more new data:")
for text, prediction in zip(new_data_extended, new_predictions_extended):
    print(f"{text} - Predicted: {prediction}")
```


## Files

1. **Dataset:** `medical_non_medical_dataset.csv`
   - This CSV file contains the cleaned and preprocessed dataset used for training and testing the models.

2. **Training Code:** `NLP_Assignment_1.ipynb`/ `nlp_assignment_1.py`
   - The Python script `NLP_Assignment_1.ipynb`/ `nlp_assignment_1.py` includes the code for fetching content from Wikipedia, cleaning the text data, and training the classification models using scikit-learn.

## Usage
- The `NLP_Assignment_1.ipynb` file works directly on Colab without the need for additional uploads. By default, it creates the dataset, facilitates easy training, and allows for result visualization.
- To use `nlp_assignment_1.py`, ensure that the required dependencies are installed first. You can do this by running:

### Requirements

- Python 3.10.12

### Dependencies

Make sure you have the following Python libraries installed:

- [pandas](https://pandas.pydata.org/): `pip install pandas`
- [Wikipedia-API](https://pypi.org/project/Wikipedia-API/): `pip install Wikipedia-API`
- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): `pip install beautifulsoup4`
- [nltk](https://www.nltk.org/): `pip install nltk`
- [scikit-learn](https://scikit-learn.org/stable/): `pip install scikit-learn`

These libraries are required to run the Python scripts and notebooks in this project.

# Update
NLP_Assignment_1_updated_version, this updated version is with the addition of a validation set to evaluate overfitting, underfitting, and misclassification for naive_bayes


