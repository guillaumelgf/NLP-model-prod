import argparse
import os
import time

import joblib

import pandas as pd
from preprocess.CustomPreProcess import CustomPreProcess
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("job_name")
    parser.add_argument("--stagging-bucket")
    args = parser.parse_args()
    return args


def get_data(url):
    df = pd.read_csv(url, parse_dates=["date"])
    df = df[~df.headline.isna()]  # Cleaning
    df = df[["headline", "category"]]
    return df


def total_accuracy(y_pred, y_true):
    return sum(y_pred == y_true) / len(y_true) * 100


def run():
    file_name = "News_Category_Dataset.csv"
    file_path = os.path.join(os.environ["DATA_PATH"], "NLP-for-fun", file_name)

    if not os.path.exists("models"):
        os.makedirs("models")

    print("Loading data: ", end="")
    start_timer = time.time()
    df = get_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"], df["category"], test_size=0.1, random_state=1
    )
    print(f"Execution {time.time() - start_timer:.2f}s")

    print("Training model", end="")
    start_timer = time.time()
    model_pipeline = Pipeline(
        steps=[
            ("pre_processing", CustomPreProcess()),
            ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=5))
        ]
    )
    model_pipeline.fit(X_train, y_train)
    print(f"Execution {time.time() - start_timer:.2f}s")

    print("Exporting model", end="")
    start_timer = time.time()
    joblib.dump(model_pipeline, "models/model.joblib")
    print(f"Execution {time.time() - start_timer:.2f}s")

    print("Evaluating", end="")
    start_timer = time.time()
    preds_test = model_pipeline.predict(X_test)
    model_acc = total_accuracy(preds_test, y_test)
    print(f"Execution {time.time() - start_timer:.2f}s")
    print(f"\nModel accuracy on test: {model_acc}%")


if __name__ == "__main__":
    # args = parse_args()
    run()
    pass
