import os
import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import json
from build import build_custom_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# --- Logger setup ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_generators(test_csv, img_size):
    logger.info(f"Loading test CSV file: {test_csv}")
    test_df = pd.read_csv(test_csv)

    ts_length = len(test_df)
    test_batch_size = max(
        sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80])
    )
    test_steps = ts_length // test_batch_size

    logger.info(f"Batch size optimal détecté : {test_batch_size} — Test steps : {test_steps}")

    test_gen = ImageDataGenerator().flow_from_dataframe(
        test_df,
        x_col='name',
        y_col='label',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        batch_size=test_batch_size
    )
    return test_gen, test_steps

def evaluate_model(valid_gen, model, output_dir):
    y_true, y_pred = [], []

    for images, labels in valid_gen:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        if len(y_true) >= valid_gen.samples:
            break

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    class_names = list(valid_gen.class_indices.keys())

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "classification_report.txt")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    class_report=classification_report(y_true, y_pred, target_names=class_names,output_dict=True)  ######
    report_json_path = os.path.join(output_dir, "classification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(class_report, f, indent=4)

    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names) )
        f.write(f"\nOverall Accuracy: {acc:.4%}\n")
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write(f"Weighted Recall: {recall:.4f}\n")
        f.write(f"Weighted F1-score: {f1:.4f}\n")

    logger.info(f"Classification report saved at : {report_path}")

    # Confusion Matrix Plot
    cf_matrix = confusion_matrix(y_true, y_pred)
    labels = [
        f'{name}\n{count}\n{pct:.2%}' 
        for name, count, pct in zip(
            ['TN','FP','FN','TP'],
            cf_matrix.flatten(),
            cf_matrix.flatten() / np.sum(cf_matrix)
        )
    ]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion Matrix saved at : {cm_path}")

    return acc, precision, recall, f1, report_path, cm_path

def run_evaluation(args):
    # --- Paramètres de base ---
    IMG_SIZE = (args.img_height, args.img_width)
    IMG_SHAPE = (args.img_height, args.img_width, 3)

    logger.info("Model Initialisation")
    model = build_custom_model(
        EfficientNetB0,
        img_shape=IMG_SHAPE,
        class_count=len(args.class_names),
        freeze_percentage=args.freeze_percentage,
        weights="imagenet",
        pooling=args.pooling,
        learning_rate=args.learning_rate,
        plot_file=os.path.join(args.output_dir, "model_plot.png"),
        show_summary=False
    )

    logger.info(f"Load weights from {args.weights}")
    model.load_weights(args.weights)

    test_gen, test_steps = create_generators(args.test_csv, IMG_SIZE)

    logger.info("Evaluation begins")
    test_loss, test_acc = model.evaluate(test_gen, steps=test_steps, verbose=1)

    logger.info(f"Test Loss: {test_loss:.4f} — Test Accuracy: {test_acc:.4%}")

    # --- MLflow Tracking ---
    mlflow.set_experiment("Pneumonia Test 1")
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_params({
            "img_height": args.img_height,
            "img_width": args.img_width,
            "learning_rate": args.learning_rate,
            "freeze_percentage": args.freeze_percentage,
            "pooling": args.pooling,
            "weights": args.weights,
            "batch_size": test_gen.batch_size,
            "test_steps": test_steps,
        })

        acc, precision, recall, f1, report_path, cm_path = evaluate_model(test_gen, model, args.output_dir)

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1
        })

        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(os.path.join(args.output_dir, "model_plot.png"))

        logger.info("Evaluation completed")

# --- Argument parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du modèle de détection de pneumonie avec MLflow")

    parser.add_argument("--test_csv", type=str, default="./data/splits/test_split.csv", help="Chemin du CSV de test")
    parser.add_argument("--weights", type=str, default="./outputs/models/Enhanced_model_V2.keras", help="Fichier de poids du modèle")
    parser.add_argument("--output_dir", type=str, default="./outputs/results", help="Répertoire de sortie des résultats")
    parser.add_argument("--img_height", type=int, default=224, help="Hauteur des images")
    parser.add_argument("--img_width", type=int, default=224, help="Largeur des images")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Taux d'apprentissage")
    parser.add_argument("--freeze_percentage", type=float, default=0.0, help="Pourcentage de gel des couches")
    parser.add_argument("--pooling", type=str, default="max", help="Type de pooling ('avg' ou 'max')")
    parser.add_argument("--class_names", nargs='+', default=['NORMAL', 'PNEUMONIA'], help="Liste des classes")
    parser.add_argument("--experiment_name", type=str, default="Pneumonia_Eval", help="Nom de l'expérience MLflow")
    parser.add_argument("--run_name", type=str, default="EfficientNetB0_Eval", help="Nom du run MLflow")

    args = parser.parse_args()
    run_evaluation(args)
