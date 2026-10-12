import pytorch_lightning as pl  # noqa: D100

from skeleplex.measurements.lumen_classifier import (
    ConvNext3ClassClassifier,
    H5DataModule,
)

# This example trains a ConvNeXt classifier to classify lumen, branch and bad slices
# Set up your training data first using the create_lumen_classifier_trainings_data.py

# Set up dataset and model
dataset_root = "dataset/"
model = ConvNext3ClassClassifier(num_classes=3)
data_module = H5DataModule(data_dir=dataset_root, batch_size=16)


# logging
logger = pl.loggers.TensorBoardLogger("logs/", name="resnet3class")

# early stopping
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=True,
    mode="min",
)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="resnet3class-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)
trainer = pl.Trainer(
    max_epochs=200,
    logger=logger,
    accelerator="gpu",
    callbacks=[early_stop_callback, checkpoint_callback],
)

# Train the model
trainer.fit(model, data_module)
