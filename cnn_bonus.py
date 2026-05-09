import tensorflow as tf
from tensorflow.keras import layers, models
import os

def load_house_dog_dataset(
    data_dir="data",
    img_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    seed=123,
):
   
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

  
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_house_dog_cnn(input_shape=(128, 128, 3)):
    """
    Simple CNN for house vs dog (binary classification).
    """
    model = models.Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=input_shape),

            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),  # output: probability of dog
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_house_dog_cnn(
    data_dir="data",
    img_size=(128, 128),
    batch_size=32,
    epochs=5,
):
    """
    Train CNN to classify images as house (0) or dog (1).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Create 'data/house' and 'data/dog' folders with images."
        )

    train_ds, val_ds = load_house_dog_dataset(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
    )

    #class names (should be ['dog', 'house'] or ['house', 'dog'])
    class_names = train_ds.class_names
    print("Class names (from folders):", class_names)
    print("Interpreting label 0 as:", class_names[0])
    print("Interpreting label 1 as:", class_names[1])

    model = build_house_dog_cnn(input_shape=img_size + (3,))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Final validation accuracy: {val_acc:.3f}")

    # Show a few predictions from validation set
    for images, labels in val_ds.take(1):
        probs = model.predict(images)
        preds = (probs >= 0.5).astype("int32").flatten()
        print("True labels:     ", labels.numpy().astype("int32").flatten()[:10])
        print("Predicted labels:", preds[:10])
        break

    return model, history


if __name__ == "__main__":
    print("CNN: House vs Dog using TensorFlow/Keras")
    data_dir = input("Enter data directory (default 'data'): ").strip() or "data"
    try:
        epochs = int(input("Enter number of epochs (e.g., 5 or 10): ").strip())
    except ValueError:
        epochs = 5

    train_house_dog_cnn(data_dir=data_dir, epochs=epochs)