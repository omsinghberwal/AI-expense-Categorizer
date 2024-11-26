# Module for training and predicting
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model(training_data):
    """
    Train a Naive Bayes model to classify expenses based on transaction descriptions.
    :param training_data: A DataFrame with 'Description' and 'Category' columns.
    :return: Trained model and vectorizer
    """
    descriptions = training_data['Description']
    categories = training_data['Category']

    # Text vectorization
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(descriptions)

    # Model training
    model = MultinomialNB()
    model.fit(X, categories)

    return model, vectorizer

def predict_categories(model, vectorizer, data):
    """
    Predict categories for new transaction descriptions.
    :param model: Trained model
    :param vectorizer: Fitted vectorizer
    :param data: DataFrame with 'Description' and 'Amount' columns
    :return: DataFrame with added 'Category' column
    """
    descriptions = data['Description']
    X = vectorizer.transform(descriptions)

    # Predict categories
    predicted_categories = model.predict(X)
    data['Category'] = predicted_categories

    return data
