import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from build import build_custom_model
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import argparse
import os
import logging
# --- Setup du logger ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def create_data_generators(image_size,batch_size):
    train_df = pd.read_csv("data/splits/train_split.csv")
    valid_df = pd.read_csv("data/splits/valid_split.csv")
    test_df = pd.read_csv("data/splits/test_split.csv")

    test_size = len(test_df)
    test_batch_size = max(sorted([test_size // n for n in range(1, test_size + 1) if test_size % n == 0 and test_size / n <= 80]))

    datagen = ImageDataGenerator()

    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='name',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    valid_generator = datagen.flow_from_dataframe(
        valid_df,
        x_col='name',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    test_generator = datagen.flow_from_dataframe(
        test_df,
        x_col='name',
        y_col='label',
        target_size=image_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        batch_size=test_batch_size
    )

    class_indices = train_generator.class_indices
    class_mapping = list(class_indices.keys())
    y_numeric = train_df['label'].map(class_indices)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array(range(len(class_mapping))), y=y_numeric)
    class_weight_dict = dict(zip(class_indices.values(), class_weights))

    # Optional adjustment (if class imbalance is critical)
    class_weight_dict[0] *= 1.5

    return train_generator, valid_generator, test_generator, class_weight_dict


def plot_training_curves(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    best_loss_epoch = np.argmin(val_loss)
    lowest_val_loss = val_loss[best_loss_epoch]
    best_acc_epoch = np.argmax(val_acc)
    highest_val_acc = val_acc[best_acc_epoch]

    epochs = list(range(1, len(train_acc) + 1))

    loss_label = f'Best epoch = {best_loss_epoch + 1} (val_loss = {lowest_val_loss:.4f})'
    acc_label = f'Best epoch = {best_acc_epoch + 1} (val_accuracy = {highest_val_acc:.4f})'

    plt.figure(figsize=(20, 8))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.scatter(best_loss_epoch + 1, lowest_val_loss, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(best_acc_epoch + 1, highest_val_acc, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def train_model(model, image_size, image_shape, epochs, batch_size, learning_rate, model_output_path, plot_output_path):
    train_gen, valid_gen, test_gen, class_weights = create_data_generators(image_size,batch_size)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_output_path, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)
    ]


    mlflow.set_experiment("Pneumonia Test 1")
    with mlflow.start_run(run_name="Model Training"):
        mlflow.tensorflow.autolog()

        mlflow.log_params({
            "image_height": image_size[0],
            "image_width": image_size[1],
            "image_shape": image_shape,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_architecture": "EfficientNetB0",
            "freeze_percentage": 0,
            "model_output_path": model_output_path,
            "plot_output_path": plot_output_path,
            "callbacks": "EarlyStopping, ModelCheckpoint, ReduceLROnPlateau",
            "class_weighted": True
        })

        logger.info("Training begins")
        history = model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=epochs,
            verbose=1,
            class_weight=class_weights,
            callbacks=callbacks
        )
        logger.info(f"Training completed and model weights saved at {model_output_path}")

        curves_path = plot_training_curves(history, save_path=plot_output_path)
        logger.info(f"Training curves saved at {curves_path}")

        mlflow.log_artifact(model_output_path)
        mlflow.log_artifact(plot_output_path)

        logger.info("Model and plots logged with MLflow")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model")
    parser.add_argument("--output_model", type=str, default="./outputs/models/Enhanced_model_V2.keras", help="Path to save the model")
    parser.add_argument("--output_plot", type=str, default="./outputs/results/training_history.png", help="Path to save the training curve")
    parser.add_argument("--output_model_plot", type=str, default="./outputs/results/model_plot.png", help="Path to save the model plot") # ..
    parser.add_argument("--epochs", type=float, default=7, help="epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--image_height", type=int, default=224, help="Height of input images")
    parser.add_argument("--image_width", type=int, default=224, help="Width of input images")
    
    args = parser.parse_args()

    image_size = (args.image_height, args.image_width)
    image_shape = (args.image_height, args.image_width, 3)
    num_classes = len(['NORMAL', 'PNEUMONIA'])  # Could be param if needed

    model = build_custom_model(
        EfficientNetB0,
        image_shape,
        num_classes,
        freeze_percentage=0,
        weights="imagenet",
        pooling="max",
        learning_rate=args.learning_rate,
        plot_file=args.output_model_plot
    )

    history = train_model(
        model=model,
        image_size=image_size,
        image_shape=image_shape,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_output_path=args.output_model,
        plot_output_path=args.output_plot
    )


if __name__ == "__main__":
    main()