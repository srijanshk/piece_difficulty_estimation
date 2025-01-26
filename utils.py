# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import partitura as pt
from Baseline_PieceDifficulty import load_piece_difficulty_dataset_notebook, compute_score_features
from yoko_ono_PieceDifficulty import compute_optimized_score_features, create_model_pipeline

train_data, _ = load_piece_difficulty_dataset_notebook("piece_difficulty_dataset/scores_difficulty_estimation")
def load_and_prepare_data_baseline():
    train_features, train_labels = [], []

    for _, (dif, fn) in train_data.items():
        try:
            score = pt.load_musicxml(fn)
            score[0].use_musical_beat()
            note_array = pt.utils.music.ensure_notearray(
                score,
                include_pitch_spelling=True,
                include_key_signature=True,
                include_time_signature=True,
                include_metrical_position=True,
                include_grace_notes=True,
            )
            features = compute_score_features(note_array)
            train_features.append(features)
            train_labels.append(dif)
        except Exception as e:
            print(f"Could not load {fn}: {e}")

    return np.array(train_features), np.array(train_labels)

def load_and_prepare_data_yoko_ono():
    train_features, train_labels = [], []

    for _, (dif, fn) in train_data.items():
        try:
            score = pt.load_musicxml(fn)
            score[0].use_musical_beat()
            note_array = pt.utils.music.ensure_notearray(
                score,
                include_pitch_spelling=True,
                include_key_signature=True,
                include_time_signature=True,
                include_metrical_position=True,
                include_grace_notes=True,
            )
            features = compute_optimized_score_features(note_array)
            train_features.append(features)
            train_labels.append(dif)
        except Exception as e:
            print(f"Could not load {fn}: {e}")

    return np.array(train_features), np.array(train_labels)

def train_and_evaluate_baseline(X, Y):
    val_size = 0.2
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, stratify=Y, random_state=42)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, Y_train)
    Y_val_pred = classifier.predict(X_val)

    acc = accuracy_score(Y_val, Y_val_pred)
    f1 = f1_score(Y_val, Y_val_pred, average="macro")
    cm = confusion_matrix(Y_val, Y_val_pred)

    return acc, f1, cm, Y_val

def train_and_evaluate_yoko_ono(X, Y):
    val_size = 0.2
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, stratify=Y, random_state=42)
    classifier = create_model_pipeline()
    classifier.fit(X_train, Y_train)
    Y_val_pred = classifier.predict(X_val)

    acc = accuracy_score(Y_val, Y_val_pred)
    f1 = f1_score(Y_val, Y_val_pred, average="macro")
    cm = confusion_matrix(Y_val, Y_val_pred)

    return acc, f1, cm, Y_val

def plot_results(baseline_scores, improved_scores, baseline_cm, improved_cm, labels, baseline_y_val, improved_y_val):
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width/2, baseline_scores, width, label='Baseline')
    plt.bar(x + width/2, improved_scores, width, label='Improved')
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Baseline vs Improved Model Comparison")
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ConfusionMatrixDisplay(baseline_cm, display_labels=np.unique(baseline_y_val)).plot(ax=ax[0], cmap='Blues')
    ax[0].set_title("Baseline Model Confusion Matrix")

    ConfusionMatrixDisplay(improved_cm, display_labels=np.unique(improved_y_val)).plot(ax=ax[1], cmap='Greens')
    ax[1].set_title("Improved Model Confusion Matrix")

    plt.tight_layout()
    plt.show()