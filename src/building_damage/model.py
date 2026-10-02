"""Later wrapper preserving the recovered U-Net architecture, loss and metrics."""

import os


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
