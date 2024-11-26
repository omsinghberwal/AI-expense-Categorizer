# Entry point of the application
import pandas as pd
from model import train_model, predict_categories
from visualization import generate_charts
import os

def main():
    print("Welcome to the AI-Powered Expense Categorizer!")
    
    # Step 1: Look for the default CSV file
    default_file = "sample_transactions.csv"
    if os.path.exists(default_file):
        print(f"Found default file: {default_file}")
        input_file = default_file
    else:
        input_file = input("Default file not found. Enter the path to your transaction CSV file: ").strip()
        if not os.path.exists(input_file):
            print("File not found. Exiting the program.")
            return

    # Step 2: Read the CSV file
    try:
        data = pd.read_csv(input_file)
        if "Description" not in data.columns or "Amount" not in data.columns:
            print("Invalid file format. Ensure it has 'Description' and 'Amount' columns.")
            return
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    print(f"File '{input_file}' loaded successfully!")

    # Step 3: Check for saved model or train a new one
    model_file = "expense_categorizer_model.pkl"
    vectorizer_file = "vectorizer.pkl"

    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        print("Loading existing model...")
        from joblib import load
        model = load(model_file)
        vectorizer = load(vectorizer_file)
    else:
        print("Training a new model...")
        training_file = input("Enter the path to your training data CSV file: ").strip()
        if not os.path.exists(training_file):
            print("Training file not found. Exiting.")
            return

        training_data = pd.read_csv(training_file)
        model, vectorizer = train_model(training_data)
        from joblib import dump
        dump(model, model_file)
        dump(vectorizer, vectorizer_file)
        print("Model trained and saved successfully!")

    # Step 4: Predict categories
    print("Categorizing expenses...")
    categorized_data = predict_categories(model, vectorizer, data)

    # Step 5: Save categorized data
    output_file = "categorized_expenses.csv"
    categorized_data.to_csv(output_file, index=False)
    print(f"Categorized data saved to '{output_file}'.")

    # Step 6: Generate visualization
    print("Generating expense distribution chart...")
    generate_charts(categorized_data)

if __name__ == "__main__":
    main()
