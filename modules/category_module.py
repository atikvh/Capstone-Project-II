"""
This handles document classification.
Supports SVM, Naive Bayes, KNN, and Random Forest for comparison.
Designed for training, evaluation, and prediction using OCR + preprocessed text.
"""

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class DocumentCategorizer:
    def __init__(self, model_type="svm", model_path="models/categorizer.pkl"):
        self.model_type = model_type.lower() # model_type: 'svm', 'nb', 'knn', or 'rf'
        self.model_path = model_path  # model_path: where to save/load trained model
        self.pipeline = None

    # Internal method to build the ML pipeline based on selected model.
    def _build_model(self):
        if self.model_type == "svm":
            model = LinearSVC()
        elif self.model_type == "nb":
            model = MultinomialNB()
        elif self.model_type == "knn":
            model = KNeighborsClassifier(n_neighbors=5)
        elif self.model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", model)
        ])

    # Train models with the dataset csv file
    def train(self, dataset_path: str, test_size=0.2):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        if "text" not in data.columns or "category" not in data.columns:
            raise ValueError("Dataset must contain 'text' and 'category' columns.")

        X = data["text"]
        y = data["category"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self._build_model()
        print(f"[Training] Model: {self.model_type.upper()} — {len(X_train)} training samples")

        self.pipeline.fit(X_train, y_train) # type: ignore
        y_pred = self.pipeline.predict(X_test) # type: ignore

        acc = accuracy_score(y_test, y_pred)
        print(f"\n[Accuracy] {self.model_type.upper()} = {acc * 100:.2f}%")
        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred))

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"[Saved] Trained model stored at: {self.model_path}")

        return acc
    # Compare all models automatically and display result summary
    def compare_models(self, dataset_path):
        models = ["svm", "nb", "knn", "rf"]
        results = {}

        print("Comparing Models (SVM, NB, KNN, RF)\n")

        for model_type in models:
            print(f"=== Training {model_type.upper()} ===")
            self.model_type = model_type
            acc = self.train(dataset_path)
            results[model_type.upper()] = acc
            print("=" * 60 + "\n")

        print("Summary of Model Accuracies:")
        for m, acc in results.items():
            print(f"{m}: {acc * 100:.2f}%")

        best_model = max(results, key=results.get) # type: ignore
        print(f"\nBest Model: {best_model} with accuracy {results[best_model] * 100:.2f}%")

    # Load pre-trained model
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        print(f"[Loaded] Model loaded from: {self.model_path}")

    # Predict category for new document text (after preprocessing module)
    def predict(self, text: str):
        if self.pipeline is None:
            self.load_model()

        if not text.strip():
            raise ValueError("Empty text provided for prediction.")

        return self.pipeline.predict([text])[0] # type: ignore


# ========== TEST BLOCK ==========
if __name__ == "__main__":
    print("Starting Categorization Module Test...\n")

    dataset_path = "datasets/replicated/training_datasets.csv"   
    ocr_output_path = "datasets/preprocess_result.txt"        

    try:
        categorizer = DocumentCategorizer(model_type="svm")

        # Option 1: Compare all models to find the best one
        categorizer.compare_models(dataset_path)

        # Option 2: Train only one model (e.g., SVM)
        # categorizer.train(dataset_path)

        # Predict with OCR text
        with open(ocr_output_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()

        categorizer.load_model()
        predicted_category = categorizer.predict(ocr_text)

        print(f"\nPredicted category for uploaded document: {predicted_category}")

    except Exception as e:
        print(f"Categorization test failed: {e}")
