import os
import pandas as pd
import mlflow
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup du logger ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def plot_label_distributions(df, output_dir="./outputs/results"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pie Chart: Label Proportion
    plt.figure(figsize=(6, 6))
    df["label"].value_counts(normalize=True).plot(
        kind="pie", autopct="%1.1f%%", textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    plt.title("Label Proportion", fontsize=14, fontweight='bold', pad=10)
    plt.ylabel("")  # Remove y-label for cleaner look
    pie_output_path = os.path.join(output_dir, "label_proportion_pie.png")
    plt.savefig(pie_output_path)
    plt.close()

    # 2. Bar Plot: Overall Label Distribution
    plt.figure(figsize=(9, 6))
    ax2 = sns.countplot(x="label", data=df, edgecolor='black', linewidth=0.5)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:  # Only annotate non-zero bars
            ax2.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.5,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    plt.title("Overall Label Distribution", fontsize=14, fontweight='bold', pad=10)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    bar_output_path = os.path.join(output_dir, "overall_label_distribution_bar.png")
    plt.savefig(bar_output_path)
    plt.close()

    return bar_output_path,pie_output_path

def preprocess_data(base_dir, output_csv):
    try:
        splits = ["test", "train", "val"]
        labels = ["NORMAL", "PNEUMONIA"]
        file_paths, file_labels = [], []

        for split in splits:
            split_dir = os.path.join(base_dir, split)
            for label in labels:
                label_dir = os.path.join(split_dir, label)
                if os.path.exists(label_dir):
                    for file_name in os.listdir(label_dir):
                        file_path = os.path.join(label_dir, file_name)
                        if os.path.isfile(file_path):
                            file_paths.append(file_path)
                            file_labels.append(label)
                else:
                    logger.warning(f"Dossier non trouvé : {label_dir}")

        df = pd.DataFrame({"name": file_paths, "label": file_labels})

        # --- Validation des données ---
        if df.empty:
            logger.error("Le dataset est vide après le prétraitement.")
            raise ValueError("Le dataset est vide.")

        if df["label"].nunique() < 2:
            logger.warning("Le dataset ne contient qu'une seule classe.")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"DataFrame sauvegardé : '{output_csv}'")

        # --- Tracking MLflow ---
        mlflow.set_experiment("Pneumonia Test 1")
        with mlflow.start_run(run_name="Preprocessing_Dataset_V1"):
            mlflow.log_param("total_images", len(df))
            mlflow.log_param("normal_count", (df["label"] == "NORMAL").sum())
            mlflow.log_param("pneumonia_count", (df["label"] == "PNEUMONIA").sum())

            bar_output_path,pie_output_path=plot_label_distributions(df)
            mlflow.log_artifact(pie_output_path)
            mlflow.log_artifact(bar_output_path)
            logger.info(f"Plots Saved on : {bar_output_path} and {pie_output_path}")
        
            mlflow.log_artifact(output_csv)
            logger.info("Données enregistrées dans MLflow.")

    except Exception as e:
        logger.exception(f"Erreur dans le prétraitement : {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des données de radiographie.")
    parser.add_argument("--base_dir", type=str, default="./chest_xray/",
                        help="Répertoire racine contenant les données 'train', 'test', 'val'")
    parser.add_argument("--output_csv", type=str, default="data/pneumonia_dataset.csv",
                        help="Chemin de sortie du fichier CSV contenant les métadonnées")

    args = parser.parse_args()
    preprocess_data(base_dir=args.base_dir, output_csv=args.output_csv)
