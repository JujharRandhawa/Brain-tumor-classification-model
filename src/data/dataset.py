"""Dataset indexing and TensorFlow input pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data.download import resolve_data_directory
from src.data.preprocessing import build_augmentation_layer

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_weights: dict[int, float]


def index_dataset(data_root: Path, config: dict) -> pd.DataFrame:
    """Build a dataframe of image paths and binary labels."""
    extensions = set(config["data"]["supported_extensions"])
    folder_map = config["data"]["class_folders"]
    records: list[dict] = []

    for label_idx, (class_name, folder_names) in enumerate(folder_map.items()):
        for folder_name in folder_names:
            class_dir = data_root / folder_name
            if not class_dir.exists():
                logger.warning("Missing folder: %s", class_dir)
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in extensions:
                    continue
                records.append(
                    {
                        "path": str(path),
                        "label": label_idx,
                        "class_name": class_name,
                    }
                )

    if not records:
        raise RuntimeError(f"No images found under {data_root}")

    df = pd.DataFrame(records)
    logger.info("Indexed %d images — class distribution:\n%s", len(df), df["class_name"].value_counts())
    return df


def split_dataset(df: pd.DataFrame, config: dict) -> DatasetSplits:
    """Stratified train/validation/test split."""
    seed = config["data"]["random_seed"]
    val_ratio = config["data"]["validation_split"]
    test_ratio = config["data"]["test_split"]

    train_df, holdout_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test,
        stratify=holdout_df["label"],
        random_state=seed,
    )

    class_weights = {}
    if config["training"]["use_class_weights"]:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_df["label"]),
            y=train_df["label"],
        )
        class_weights = {int(c): float(w) for c, w in zip(np.unique(train_df["label"]), weights)}

    return DatasetSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        class_weights=class_weights,
    )


def create_data_pipeline(
    dataframe: pd.DataFrame,
    config: dict,
    training: bool = False,
    shuffle: bool = True,
):
    """Build a tf.data.Dataset from a split dataframe."""
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    image_size = tuple(config["data"]["image_size"])
    batch_size = config["training"]["batch_size"]
    aug = build_augmentation_layer(config) if training else None

    paths = dataframe["path"].tolist()
    labels = dataframe["label"].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=config["data"]["random_seed"])

    def _map_fn(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        if aug is not None:
            image = aug(image)
        image = preprocess_input(image)
        return image, label

    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_datasets(config: dict) -> tuple[DatasetSplits, object, object, object]:
    """Resolve data path, split, and return train/val/test pipelines."""
    data_root = resolve_data_directory(config["paths"]["raw_data_dir"])
    df = index_dataset(data_root, config)
    splits = split_dataset(df, config)

    train_ds = create_data_pipeline(splits.train, config, training=True, shuffle=True)
    val_ds = create_data_pipeline(splits.val, config, training=False, shuffle=False)
    test_ds = create_data_pipeline(splits.test, config, training=False, shuffle=False)

    return splits, train_ds, val_ds, test_ds
