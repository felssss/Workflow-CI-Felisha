import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data():
    train_df = pd.read_csv("namadataset_preprocessing/train.csv")
    test_df = pd.read_csv("namadataset_preprocessing/test.csv")
    return train_df, test_df

def train_model(X, y):
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def main():
    train_df, test_df = load_data()

    X_train = train_df.drop(columns=["Survived"])
    y_train = train_df["Survived"]
    X_test = test_df.drop(columns=["Survived"])
    y_test = test_df["Survived"]

    model = train_model(X_train, y_train)
    acc = evaluate(model, X_test, y_test)

    print(f"Training finished. Accuracy: {acc}")

if __name__ == "__main__":
    main()