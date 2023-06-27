import copy
import os
import re
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import efficient.cflags
import efficient.datasets
import efficient.effnetv2_configs
import efficient.effnetv2_model
import efficient.hparams
import efficient.utils as utils


img_height = img_width = 224
eval_data_dir=   "Data/Test"
batch_size = 32
check = sys.argv[1]+".h5"

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(
    eval_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')
test_steps=validation_generator.__len__()


tf.keras.backend.clear_session()
if sys.argv[2] != "efficient":
    if sys.argv[2] == "mobile":
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet"
        )
    elif sys.argv[2] == 'densenet':
        base_model = tf.keras.applications.DenseNet169(
            include_top=False,
            weights="imagenet"
        )
    else:
        base_model = tf.keras.applications.ResNet101V2(
            include_top=False,
            weights="imagenet"
        )
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, outputs)
    _ = model(tf.ones([1, 224, 224, 3]), training=False)
else:
    model = effnetv2_model.EffNetV2Model(model_name='efficientnetv2-m')
    _ = model(tf.ones([1, 224, 224, 3]), training=False)
    ckpt = 'efficientnetv2-m'
    # utils.restore_tf2_ckpt(model, ckpt, exclude_layers=('_fc', 'optimizer'))

model.load_weights(check)

config = copy.deepcopy(efficient.hparams.base_config)
config.train.lr_base=0.001
scaled_lr = config.train.lr_base * (config.train.batch_size / 256.0)
scaled_lr_min = config.train.lr_min * (config.train.batch_size / 256.0)
learning_rate = utils.WarmupLearningRateSchedule(
        scaled_lr,
        steps_per_epoch=1000,
        decay_epochs=config.train.lr_decay_epoch,
        warmup_epochs=config.train.lr_warmup_epoch,
        decay_factor=config.train.lr_decay_factor,
        lr_decay_type=config.train.lr_sched,
        total_steps=1000*50,
        minimal_lr=scaled_lr_min)

optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate,decay=0.9,
                        epsilon=0.001,
                        momentum=0.9)


model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.0, from_logits=True),
    metrics=[
        tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc_top1'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='acc_top3')
    ],
)

true_labels = validation_generator.classes
Out = model.predict(validation_generator)
y_pred = np.array([np.argmax(x) for x in Out])
CM = tf.math.confusion_matrix(true_labels,y_pred,num_classes=10)
print(CM)
ex = model.evaluate(validation_generator, verbose=2,return_dict=True)
print(ex)

