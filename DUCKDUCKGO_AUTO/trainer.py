import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.7,1.3)
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'data/train', target_size=(128,128), batch_size=32, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    'data/val',   target_size=(128,128), batch_size=32, class_mode='binary'
)

# 2. Base model (pre-trained)
base = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False   # freeze for initial training

# 3. Add classification head
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# 5. (Optional) fine‑tune lower layers
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(train_gen, validation_data=val_gen, epochs=3)

# 6. Save for inference
model.save('duck_classifier.h5')
