import random

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from preprocessing import load_and_process


def perform_grid_search_cv(
    model: ClassifierMixin,
    X: csr_matrix,
    y: np.ndarray,
    param_grid: dict[str, list[float | int | str]],
    cv: int = 5,
) -> tuple[dict, float, float]:
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        scoring=["accuracy", "f1_weighted"],
        refit="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    best_f1 = grid_search.cv_results_["mean_test_f1_weighted"][grid_search.best_index_]
    return best_params, best_accuracy, best_f1


def compare_models_with_grid_search_cv(
    X: csr_matrix,
    y: np.ndarray,
    model_params: list[dict],
    models: list[ClassifierMixin],
    model_names: list[str],
    save_path: str = "../data/grid_searchs/grid_search_results.csv",
) -> pd.DataFrame:
    """Perform grid search with cross validation on different models and return the results in a DataFrame."""
    results = []
    for model, model_name, param_grid in zip(
        models, (progress_bar := tqdm(model_names)), model_params
    ):
        progress_bar.set_postfix_str(model_name)
        best_params, best_accuracy, best_f1 = perform_grid_search_cv(
            model, X, y, param_grid, cv=3
        )
        results.append(
            {
                "Model": model_name,
                "Accuracy": best_accuracy,
                "F1-score": best_f1,
                "Best Parameters": best_params,
            }
        )

    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    return results_df


def run_grid_search(save_path: str = None) -> pd.DataFrame:
    # Define the parameter grids
    sgd_params = {
        "loss": ["hinge", "log_loss", "squared_hinge", "modified_huber"],
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    logistic_reg_params = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }

    ridge_params = {"alpha": [0.01, 0.1, 1.0, 10.0]}

    extra_trees_params = {
        "n_estimators": [10, 50, 100, 200],
        "max_depth": [None, 10, 30, 50],
    }

    knn_params = {
        "n_neighbors": list(range(1, 10, 1)),
        "weights": ["uniform", "distance"],
    }

    model_params = [
        sgd_params,
        logistic_reg_params,
        ridge_params,
        extra_trees_params,
        knn_params,
    ]

    models = [
        SGDClassifier(),
        LogisticRegression(),
        RidgeClassifier(),
        ExtraTreesClassifier(),
        KNeighborsClassifier(),
    ]

    model_names = [
        "SGDClassifier",
        "LogisticRegression",
        "RidgeClassifier",
        "ExtraTreesClassifier",
        "KNeighborsClassifier",
    ]

    embedding_method, tokenize_sentences = (
        "tf-idf",
        False,
    )
    X, y = load_and_process(
        "../data/train.json",
        embedding_method=embedding_method,
        train_mode=True,
        tokenize_sentences=tokenize_sentences,
    )

    if not save_path:
        if tokenize_sentences:
            save_path = (
                f"../data/grid_searchs/{embedding_method}/results_with_tokenization.csv"
            )
        else:
            save_path = f"../data/grid_searchs/{embedding_method}/results_without_tokenization.csv"

    grid_search_results = compare_models_with_grid_search_cv(
        X, y, model_params, models, model_names, save_path=save_path
    )

    return grid_search_results


if __name__ == "__main__":
    # Fixing randomness to get reproducible results
    SEED_VAL = 42
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    run_grid_search()
