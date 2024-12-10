import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define generator function
def generator(dir, gen=ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24, 24), class_mode='binary'):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )

# Batch size and target size
BS = 32
TS = (24, 24)

# Augmented training data
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_batch = train_gen.flow_from_directory(
    'data/train',
    shuffle=True,
    batch_size=BS,
    target_size=TS,
    color_mode='grayscale',
    class_mode='binary'
)
valid_batch = generator('data/valid', shuffle=True, batch_size=BS, target_size=TS)

# Debugging: Print class indices and sample counts
print("Class indices:", train_batch.class_indices)
print("Number of training samples:", train_batch.samples)
print("Number of validation samples:", valid_batch.samples)

# Dynamically calculate steps per epoch
SPE = train_batch.samples // BS
VS = valid_batch.samples // BS

# Simplified model architecture
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_loss')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

# Train the model
history = model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=15,
    steps_per_epoch=SPE,
    validation_steps=VS,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Save the final model
os.makedirs('models', exist_ok=True)
model.save('models/final_model.keras')

# Plot training performance
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Evaluate the model on validation data
loss, accuracy = model.evaluate(valid_batch)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")