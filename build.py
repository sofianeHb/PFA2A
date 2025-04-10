import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import EfficientNetB0
import json

def build_custom_model(
    base_model_class,
    img_shape,
    class_count,
    freeze_percentage=0.8,
    weights="imagenet",
    pooling="max",
    learning_rate=0.0001,
    plot_file="model_plot.png",
    show_summary=True
):
    """
    Build and compile a custom model using any pre-trained base from Keras Applications.

    Args:
        base_model_class: Keras Applications model class (e.g., tf.keras.applications.EfficientNetB0).
        img_shape: Tuple of input image shape (height, width, channels).
        class_count: Number of output classes.
        freeze_percentage: Fraction of base model layers to freeze (0 to 1, default 0.8).
        weights: Pre-trained weights to use (default "imagenet").
        pooling: Pooling method for base model (e.g., 'avg', 'max', None).
        learning_rate: Learning rate for optimizer (default 0.00001).
        plot_file: File path to save model plot (default "model_plot.png").

    Returns:
        Compiled Keras Model.
    """
    # Define input tensor
    inputs = tf.keras.Input(shape=img_shape)

    # Create pre-trained base model
    base_model = base_model_class(
        include_top=False,
        weights=weights,
        input_shape=img_shape,
        pooling=pooling
    )

    # Freeze a percentage of layers
    total_layers = len(base_model.layers)
    num_freeze = int(total_layers * freeze_percentage)
    for layer in base_model.layers[:num_freeze]:
        layer.trainable = False
    for layer in base_model.layers[num_freeze:]:
        layer.trainable = True

    # Pass input through base model
    x = base_model(inputs)

    # Add custom layers
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01),
                     activity_regularizer=regularizers.l1(0.003),
                     bias_regularizer=regularizers.l1(0.003), activation='relu')(x)
    x = layers.Dropout(rate=0.3, seed=123)(x)
    x = layers.Dense(class_count, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    if show_summary:
        # Display summary and save plot
        model.summary()
        plot_model(
            model,
            to_file=plot_file,
            show_shapes=True
        )
        print(f"Model plot saved as '{plot_file}'")

    return model

if __name__ == "__main__":
    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']  # Example class labels pneumonia
    CLASS_COUNT = len(CLASS_NAMES)
    H, W = 224, 224
    IMG_SIZE = (H, W)                      # Standard input size for pretrained models
    IMG_SHAPE = (H, W, 3)
    LEARNING_RATE = 0.0001    
    model= build_custom_model(
        EfficientNetB0,
        IMG_SHAPE,
        CLASS_COUNT,
        freeze_percentage=0,
        weights="imagenet",
        pooling="max",
        learning_rate=LEARNING_RATE,
        plot_file="model_plot.png"
    )
    #model.save("model.keras")