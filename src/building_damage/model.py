"""Later wrapper preserving the recovered U-Net architecture, loss and metrics."""

import os
import random


def seed_framework(tf, seed):
    """Seed a fresh process without tf-keras 2.16's Python 3.12 randint bug.

    keras.utils.set_random_seed initializes a legacy generator whose initializer
    calls randint(1, 1e9), rejected by Python 3.12. Public library seed setters
    below avoid that path; cross-device bitwise determinism is not promised.
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def framework():
    # Legacy selection must happen before either library initializes Keras.
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import tensorflow as tf
    import segmentation_models as sm

    sm.set_framework("tf.keras")
    return tf, sm


def build_model(config, *, training=False):
    tf, sm = framework()
    model = sm.Unet(
        config["backbone"],
        classes=config["classes"],
        input_shape=(config["image_size"], config["image_size"], 3),
        activation="softmax",
        encoder_weights=config["encoder_weights"] if training else None,
        encoder_freeze=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss=sm.losses.DiceLoss(),
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
    )
    return model
