# ------------------------------------------------------------
# Project : Industrial Surface Crack Detection using CNN
# Author  : Vedant Dhamal
# ------------------------------------------------------------

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------
# Step 1: Basic Configuration
# ------------------------------------------------------------

print("=" * 70)
print("        Marvellous Infosystems")
print("        Industrial Surface Crack Detection using CNN")
print("=" * 70)

ORIGINAL_DATASET = "CrackDataset"

POSITIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Positive")
NEGATIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Negative")

PROCESSED_DATASET = "Processed_CrackDataset"

# Standard input size for every image supplied to the CNN.
IMAGE_SIZE = 128
# Number of images processed in one training batch.
BATCH_SIZE = 32
# Maximum number of complete passes through the training dataset.
EPOCHS = 15
# Fixed seed improves reproducibility of random operations.
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ------------------------------------------------------------
# Step 2: Check Original Dataset
# ------------------------------------------------------------

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        print("ERROR: Folder not found:", folder_path)
        exit()


check_folder(POSITIVE_FOLDER)
check_folder(NEGATIVE_FOLDER)


def get_image_files(folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        file for file in os.listdir(folder)
        if file.lower().endswith(valid_extensions)
    ]


positive_images = get_image_files(POSITIVE_FOLDER)
negative_images = get_image_files(NEGATIVE_FOLDER)

print("Original Positive Images:", len(positive_images))
print("Original Negative Images:", len(negative_images))

if len(positive_images) == 0 or len(negative_images) == 0:
    print("ERROR: Positive or Negative folder contains no images.")
    exit()


# ------------------------------------------------------------
# Step 3: Create Industrial Folder Structure
# ------------------------------------------------------------

folders = [
    "train/Crack",
    "train/NoCrack",
    "validation/Crack",
    "validation/NoCrack",
    "test/Crack",
    "test/NoCrack"
]

# Remove any previously generated split so old images cannot remain in
# train/validation/test folders when the program is executed again.
if os.path.exists(PROCESSED_DATASET):
    shutil.rmtree(PROCESSED_DATASET)

# Create a fresh train/validation/test directory structure.
for folder in folders:
    os.makedirs(os.path.join(PROCESSED_DATASET, folder), exist_ok=True)


# ------------------------------------------------------------
# Step 4: Split Data into Train, Validation, Test
# ------------------------------------------------------------

def split_and_copy_images(source_folder, image_files, class_name):
    random.shuffle(image_files)

    total_images = len(image_files)

    # 70% training, 15% validation and the remaining ~15% testing.
    train_count = int(total_images * 0.70)
    validation_count = int(total_images * 0.15)

    train_files = image_files[:train_count]
    validation_files = image_files[train_count:train_count + validation_count]
    test_files = image_files[train_count + validation_count:]

    split_data = {
        "train": train_files,
        "validation": validation_files,
        "test": test_files
    }

    for split_name, files in split_data.items():
        destination_folder = os.path.join(
            PROCESSED_DATASET,
            split_name,
            class_name
        )

        for file in files:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)

            if not os.path.exists(destination_path):
                shutil.copy(source_path, destination_path)

    print(f"{class_name} Images Split:")
    print("Training Images   :", len(train_files))
    print("Validation Images :", len(validation_files))
    print("Testing Images    :", len(test_files))
    print("-" * 70)


split_and_copy_images(POSITIVE_FOLDER, positive_images, "Crack")
split_and_copy_images(NEGATIVE_FOLDER, negative_images, "NoCrack")


# ------------------------------------------------------------
# Step 5: Define Train, Validation and Test Paths
# ------------------------------------------------------------

train_dir = os.path.join(PROCESSED_DATASET, "train")
validation_dir = os.path.join(PROCESSED_DATASET, "validation")
test_dir = os.path.join(PROCESSED_DATASET, "test")


# ------------------------------------------------------------
# Step 6: Image Preprocessing and Augmentation
# ------------------------------------------------------------

# Normalize training pixels and apply augmentation to improve generalization.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,       # Pixel values: 0..255 -> approximately 0..1
    rotation_range=15,       # Random rotation up to approximately +/-15 degrees
    zoom_range=0.2,          # Random zoom variation
    width_shift_range=0.1,   # Horizontal shift up to 10% of image width
    height_shift_range=0.1,  # Vertical shift up to 10% of image height
    horizontal_flip=True     # Random left-right flip
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",  # Binary labels because there are two classes
    shuffle=False           # Preserve order for confusion-matrix comparison
)

print("Class Indices:", train_data.class_indices)


# ------------------------------------------------------------
# Step 7: Display Sample Images
# ------------------------------------------------------------

sample_images, sample_labels = next(train_data)

plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(sample_images[i])

    if sample_labels[i] == train_data.class_indices["Crack"]:
        plt.title("Crack")
    else:
        plt.title("No Crack")

    plt.axis("off")

plt.suptitle("Marvellous CNN Sample Training Images")
plt.show()


# ------------------------------------------------------------
# Step 8: Build Industrial CNN Model
# ------------------------------------------------------------

# Sequential connects layers one after another.
model = Sequential()

# Convolution block 1:
# 32 learnable filters, each 3x3; ReLU adds non-linearity.
# Input shape is 128 x 128 x 3 (RGB).
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(BatchNormalization())            # Stabilize intermediate activations
model.add(MaxPooling2D(pool_size=(2, 2)))  # Downsample spatial dimensions

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convert multidimensional feature maps to a one-dimensional vector.
model.add(Flatten())

# Fully connected classification layers.
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))  # Drop 50% of outputs during training to reduce overfitting

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))  # Additional regularization

# One sigmoid output neuron for binary classification.
# The result is a probability-like value between 0 and 1 for class index 1.
model.add(Dense(1, activation="sigmoid"))


# ------------------------------------------------------------
# Step 9: Compile Model
# ------------------------------------------------------------

# Adam updates weights from gradients.
# Binary crossentropy is the loss for this sigmoid binary classifier.
# Accuracy is tracked as the primary displayed metric.
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ------------------------------------------------------------
# Step 10: Industrial Training Callbacks
# ------------------------------------------------------------

# Stop training when validation loss stops improving.
# restore_best_weights returns the in-memory model to its best val_loss weights.
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# Save the model whenever validation accuracy reaches a new best value.
checkpoint = ModelCheckpoint(
    "Best_Crack_Detection_Model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Reduce the learning rate when validation loss reaches a plateau.
# factor=0.2 means: new learning rate = current learning rate * 0.2.
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=0.00001,
    verbose=1
)


# ------------------------------------------------------------
# Step 11: Train CNN Model
# ------------------------------------------------------------

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data,
    callbacks=[early_stop, checkpoint, reduce_lr]
)


# ------------------------------------------------------------
# Step 12: Plot Accuracy Graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Marvellous CNN Training vs Validation Accuracy")
plt.legend()
plt.show()


# ------------------------------------------------------------
# Step 13: Plot Loss Graph
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Marvellous CNN Training vs Validation Loss")
plt.legend()
plt.show()


# ------------------------------------------------------------
# Step 14: Evaluate Model on Test Dataset
# ------------------------------------------------------------

print("=" * 70)
print("Testing Model on Unseen Test Data")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy * 100)


# ------------------------------------------------------------
# Step 15: Confusion Matrix and Classification Report
# ------------------------------------------------------------

# Generate sigmoid outputs and convert them to class indices using 0.5.
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

actual_classes = test_data.classes

print("Confusion Matrix:")
print(confusion_matrix(actual_classes, predicted_classes))

print("Classification Report:")
print(classification_report(
    actual_classes,
    predicted_classes,
    target_names=list(test_data.class_indices.keys())
))


# ------------------------------------------------------------
# Step 16: Save Final Model
# ------------------------------------------------------------

model.save("Final_Marvellous_Crack_Detection_Model.keras")

print("Final model saved successfully.")


# ------------------------------------------------------------
# Step 17: Single Image Prediction Function
# ------------------------------------------------------------

def predict_single_image(image_path):
    if not os.path.exists(image_path):
        print("ERROR: Image not found:", image_path)
        return

    # Apply the same basic preprocessing expected by the trained model.
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)                  # Image -> numerical array
    img_array = img_array / 255.0                  # Normalize to approximately 0..1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    prediction_value = prediction[0][0]

    crack_index = train_data.class_indices["Crack"]

    if crack_index == 1:
        final_result = "Crack Detected" if prediction_value > 0.5 else "No Crack"
    else:
        final_result = "No Crack" if prediction_value > 0.5 else "Crack Detected"

    print("=" * 70)
    print("Single Image Prediction")
    print("=" * 70)
    print("Image Path       :", image_path)
    print("Prediction Value :", prediction_value)
    print("Final Result     :", final_result)

    plt.imshow(load_img(image_path))
    plt.title(final_result)
    plt.axis("off")
    plt.show()


# ------------------------------------------------------------
# Step 18: Test Single Image
# ------------------------------------------------------------

# Change image name according to your actual image file
sample_test_image = os.path.join(PROCESSED_DATASET, "test", "Crack")

test_images = get_image_files(sample_test_image)

if len(test_images) > 0:
    predict_single_image(os.path.join(sample_test_image, test_images[0]))
else:
    print("No test image found for single image prediction.")