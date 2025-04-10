import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os
import argparse
import logging
import matplotlib.pyplot as plt

# --- Logging structuré ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)





import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import os

def plot_class_distribution(train_df, valid_df, test_df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    counts = {
        "train": train_df["label"].value_counts(),
        "valid": valid_df["label"].value_counts(),
        "test": test_df["label"].value_counts()
    }

    labels = ["NORMAL", "PNEUMONIA"]
    x = range(len(labels))
    bar_width = 0.25

    # Bar positions & values
    train_pos = [p - bar_width for p in x]
    valid_pos = list(x)
    test_pos = [p + bar_width for p in x]

    train_vals = [counts["train"].get(l, 0) for l in labels]
    valid_vals = [counts["valid"].get(l, 0) for l in labels]
    test_vals = [counts["test"].get(l, 0) for l in labels]

    # 1. 🥧 Pie Chart: Dataset Split
    plt.figure(figsize=(6, 6))
    sizes = [len(train_df), len(valid_df), len(test_df)]
    pie_labels = ['Training Set', 'Validation Set', 'Test Set']
    plt.pie(
        sizes,
        labels=pie_labels,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    plt.title('Dataset Split', fontsize=14, fontweight='bold', pad=10)
    plt.axis('equal')  # Circle
    pie_output_path = os.path.join(output_dir, "dataset_split_pie.png")
    plt.savefig(pie_output_path)
    plt.close()

    # 2. 📊 Bar Chart: Class distribution per split
    plt.figure(figsize=(9, 6))
    plt.bar(train_pos, train_vals, width=bar_width, label="Train", color="skyblue")
    plt.bar(valid_pos, valid_vals, width=bar_width, label="Valid", color="orange")
    plt.bar(test_pos, test_vals, width=bar_width, label="Test", color="green")

    # Add numbers on top of each bar
    for pos, val in zip(train_pos, train_vals):
        plt.text(pos, val + 3, str(val), ha='center', va='bottom', fontsize=9)
    for pos, val in zip(valid_pos, valid_vals):
        plt.text(pos, val + 3, str(val), ha='center', va='bottom', fontsize=9)
    for pos, val in zip(test_pos, test_vals):
        plt.text(pos, val + 3, str(val), ha='center', va='bottom', fontsize=9)

    plt.xticks(x, labels)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution by Split", fontsize=12)
    plt.legend()
    plt.tight_layout()
    bar_output_path = os.path.join(output_dir, "class_distribution_bar.png")
    plt.savefig(bar_output_path)
    plt.close()

    return pie_output_path, bar_output_path




def log_class_distribution(df, split_name):
    pneumonia_count = (df["label"] == "PNEUMONIA").sum()
    normal_count = (df["label"] == "NORMAL").sum()
    mlflow.log_param(f"{split_name}_pneumonia_count", pneumonia_count)
    mlflow.log_param(f"{split_name}_normal_count", normal_count)
    logger.info(f"{split_name} => PNEUMONIA: {pneumonia_count}, NORMAL: {normal_count}")

def split_data(input_csv, output_dir):
    try:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Fichier introuvable : {input_csv}")
        
        df = pd.read_csv(input_csv)
        if df.empty:
            raise ValueError("Le dataset est vide.")

        # Mélange du dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split stratifié
        train_df, temp_df = train_test_split(df, test_size=0.35, stratify=df["label"], random_state=42)
        valid_df, test_df = train_test_split(temp_df, test_size=0.4, stratify=temp_df["label"], random_state=42)

        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train_split.csv")
        valid_path = os.path.join(output_dir, "valid_split.csv")
        test_path = os.path.join(output_dir, "test_split.csv")

        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Données sauvegardées :\nTrain: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

        # --- Tracking MLflow ---
        mlflow.set_experiment("Pneumonia Test 1")
        with mlflow.start_run(run_name="Split Data"):
            mlflow.log_param("train_size", len(train_df))
            mlflow.log_param("valid_size", len(valid_df))
            mlflow.log_param("test_size", len(test_df))

            # 🔍 Distribution par split
            log_class_distribution(train_df, "train")
            log_class_distribution(valid_df, "valid")
            log_class_distribution(test_df, "test")
            # 🔍 Graphique de distribution
            pie_output_path, bar_output_path = plot_class_distribution(train_df, valid_df, test_df)
            mlflow.log_artifact(pie_output_path)
            mlflow.log_artifact(bar_output_path)
            logger.info(f"Graphiques sauvegardés dans : {bar_output_path} et {pie_output_path}")


            mlflow.log_artifacts(output_dir)
            logger.info("Infos de split enregistrées dans MLflow.")

    except Exception as e:
        logger.exception(f"Erreur pendant le split : {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split du dataset en train/valid/test.")
    parser.add_argument("--input_csv", type=str, default="data/pneumonia_dataset.csv",
                        help="Fichier CSV contenant les métadonnées des images")
    parser.add_argument("--output_dir", type=str, default="data/splits",
                        help="Répertoire de sortie pour les fichiers split")

    args = parser.parse_args()
    split_data(input_csv=args.input_csv, output_dir=args.output_dir)
