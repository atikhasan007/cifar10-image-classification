import tensorflow as tf
import numpy as np
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_data(self):

        data_path = Path(self.config.training_data)

        self.X_train = np.load(data_path / "X_train.npy")
        self.X_test = np.load(data_path / "X_test.npy")
        self.y_train = np.load(data_path / "y_train.npy")
        self.y_test = np.load(data_path / "y_test.npy")

    def get_base_model(self):

        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def get_callbacks(self):

        return [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_learning_rate,
                verbose=1
            ),

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]

    def train(self):

    self.load_data()
    self.get_base_model()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.config.learning_rate
    )

    self.model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    if self.config.is_augmentation:

        train_datagen = ImageDataGenerator(
            rotation_range=self.config.rotation_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            horizontal_flip=self.config.horizontal_flip,
            zoom_range=self.config.zoom_range,
            brightness_range=self.config.brightness_range,
            shear_range=self.config.shear_range,
            channel_shift_range=self.config.channel_shift_range,
            validation_split=self.config.validation_split
        )

        train_generator = train_datagen.flow(
            self.X_train,
            self.y_train,
            batch_size=self.config.batch_size,
            subset="training",
            shuffle=True
        )

        validation_generator = train_datagen.flow(
            self.X_train,
            self.y_train,
            batch_size=self.config.batch_size,
            subset="validation",
            shuffle=False
        )

        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.config.epochs,
            callbacks=self.get_callbacks(),
            verbose=2
        )

    else:

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=self.config.validation_split,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=self.get_callbacks(),
            verbose=2
        )

    self.model.save(self.config.trained_model_path)

    return history
