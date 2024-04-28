import os
import string
from typing import Literal

import numpy as np
import pandas as pd
import spacy
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils import ObjectManager, load_data


class Lemmatizer:
    def __init__(
        self,
        nlp: spacy.Language = None,
        stop_words: set[str] = None,
        punctuations: str = None,
    ) -> None:
        if not nlp:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp
        if not stop_words:
            self.stop_words = self.nlp.Defaults.stop_words
        else:
            self.stop_words = stop_words
        if not punctuations:
            self.punctuations = string.punctuation
        else:
            self.punctuations = punctuations

    def lemmatize(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        mytokens = [word.lemma_.lower().strip() for word in doc]
        mytokens = [
            word
            for word in mytokens
            if word not in self.stop_words and word not in self.punctuations
        ]
        sentence = " ".join(mytokens)
        return sentence


class TfidfTransformer:
    def __init__(self) -> None:
        self.tfidf_vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.array) -> None:
        X = self.tfidf_vectorizer.fit_transform(X).toarray()
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, X: np.array) -> None:
        X = self.tfidf_vectorizer.transform(X).toarray()
        X = self.scaler.transform(X)
        return X


def tfidf_transformer(
    data: pd.DataFrame, save_path: str, train_mode: bool, column_to_embed: str
) -> tuple[csr_matrix, np.ndarray]:
    if train_mode:
        tfidf_transformer = TfidfTransformer()
        X = csr_matrix(tfidf_transformer.fit_transform(data[column_to_embed].values))
        y = data["label"].values
        tfidf_manager = ObjectManager(save_path)
        tfidf_manager.save_object(tfidf_transformer)
    else:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f'Could not find "{save_path}".')
        tfidf_manager = ObjectManager(save_path)
        tfidf_transformer: TfidfTransformer = tfidf_manager.load_object()
        X = csr_matrix(tfidf_transformer.transform(data[column_to_embed].values))
        y = data["label"].values
    return X, y


class StTransformer:
    def __init__(self) -> None:
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.array) -> None:
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, X: np.array) -> None:
        X = self.scaler.transform(X)
        return X


def st_transformer(
    data: pd.DataFrame, save_path: str, train_mode: bool, column_to_embed: str
) -> tuple[csr_matrix, np.ndarray]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    data["embeddings"] = data[column_to_embed].apply(model.encode)
    if train_mode:
        st_transformer = StTransformer()
        X = csr_matrix(st_transformer.fit_transform(data["embeddings"].to_list()))
        y = data["label"].values
        st_manager = ObjectManager(save_path)
        st_manager.save_object(st_transformer)
    else:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f'Could not find "{save_path}".')
        st_manager = ObjectManager(save_path)
        st_transformer: StTransformer = st_manager.load_object()
        X = csr_matrix(st_transformer.transform(data["embeddings"].to_list()))
        y = data["label"].values
    return X, y


def load_and_process(
    file_path: str,
    embedding_method: Literal["tf-idf", "sentence-transformer"],
    train_mode: bool,
    tokenize_sentences: bool = True,
    models_path: str = "../data/models/",
    transform_labels: bool = False,
) -> tuple[csr_matrix, np.ndarray]:
    data = load_data(file_path, transform_labels=transform_labels)

    column_to_embed = "text"
    if tokenize_sentences:
        tokenizer = Lemmatizer()
        data["tokenized"] = data[column_to_embed].apply(tokenizer.lemmatize)
        column_to_embed = "tokenized"

    if embedding_method == "tf-idf":
        tfidf_save_path = os.path.join(models_path, "tfidf.pkl")
        X, y = tfidf_transformer(data, tfidf_save_path, train_mode, column_to_embed)

    if embedding_method == "sentence-transformer":
        st_save_path = os.path.join(models_path, "sentence-transformer.pkl")
        X, y = st_transformer(data, st_save_path, train_mode, column_to_embed)

    if embedding_method not in {"tf-idf", "sentence-transformer"}:
        raise ValueError(
            f"Choose tf-idf or sentence-transformer as embedding method, not {embedding_method}."
        )

    return X, y
