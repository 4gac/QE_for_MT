import argparse

from keras.layers import Dense, Input
from keras.models import Sequential

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


def build_model(input_shape):
    model = Sequential(
        [
            Input(shape=(input_shape,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
    )
    return model


def run(verbose: int):
    file_path = "qe_dataset.tsv"
    start_row = 2000
    skiprows = range(1, start_row)  # skip 1st 2000 rows

    data = pd.read_csv(file_path, sep="\t", skiprows=skiprows)

    features = data[
        [
            # adequacy feats
            "SBERT",
            # complexity feats
            "TOKENS",
            "DEPTH",
            "ADD",
            "INTER",
            # fluency feats
            "PARETO_NUM",
        ]
    ]
    labels = data["USABILITY"]  # human translation quality label

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # build model
    model = build_model(X_train_scaled.shape[1])
    model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        verbose=verbose,
    )

    # Save the model and scaler
    model.save("my_model.keras")
    joblib.dump(scaler, "my_scaler.pkl")

    # Predict on the test set
    predictions = model.predict(X_test_scaled).flatten()

    # Calculate correlations
    pearson_corr, _ = pearsonr(y_test, predictions)
    spearman_corr, _ = spearmanr(y_test, predictions)

    print("Results")
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Spearman Correlation: {spearman_corr}")


def run_k_fold(verbose: int):
    file_path = "qe_dataset.tsv"
    start_row = 2000
    skiprows = range(1, start_row)

    data = pd.read_csv(file_path, sep="\t", skiprows=skiprows)

    features = data[
        [
            "SBERT",
            "TOKENS",
            "DEPTH",
            "ADD",
            "INTER",
            "PARETO_NUM",
        ]
    ]
    labels = data["USABILITY"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pearson_scores = []
    spearman_scores = []

    fold = 1
    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = build_model(X_train_scaled.shape[1])
        model.fit(
            X_train_scaled,
            y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            verbose=verbose,
        )

        predictions = model.predict(X_test_scaled).flatten()

        pearson_corr, _ = pearsonr(y_test, predictions)
        spearman_corr, _ = spearmanr(y_test, predictions)

        pearson_scores.append(pearson_corr)
        spearman_scores.append(spearman_corr)

        # Save the model and scaler for each fold
        model_path = f"model_fold_{fold}.keras"
        scaler_path = f"scaler_fold_{fold}.pkl"
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        fold += 1

    avg_pearson = np.mean(pearson_scores)
    avg_spearman = np.mean(spearman_scores)

    print("K-fold Results:")
    print(f"Average Pearson Correlation: {avg_pearson}")
    print(f"Average Spearman Correlation: {avg_spearman}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the model.")
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbose level for model training"
    )
    args = parser.parse_args()
    run_k_fold(args.verbose)
    run(args.verbose)


if __name__ == "__main__":
    main()
