import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr
import cv2
from PIL import Image

# --- SETTINGS ---
dataset_path = "dataset/modified-dataset"
img_size = (224, 224)
batch_size = 32
epochs = 20
model_filename = "e_waste_model.h5"

# --- DATA PREPROCESSING ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())

# --- MODEL BUILDING ---
if os.path.exists(model_filename):
    model = load_model(model_filename)
    print("Loaded saved model.")
else:
    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    model.save(model_filename)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Evaluation
    val_generator.reset()
    Y_pred = model.predict(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = val_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# --- GRAD-CAM FUNCTION ---
def generate_gradcam(model, img_array, class_index):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-3).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

# --- GRADIO INTERFACE ---
def predict_with_cam(img):
    img_resized = img.resize(img_size)
    img_array = preprocess_input(np.expand_dims(np.array(img_resized), axis=0))
    preds = model.predict(img_array)[0]
    top_class = np.argmax(preds)
    heatmap = generate_gradcam(model, img_array, top_class)
    
    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)

    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}, overlay

interface = gr.Interface(
    fn=predict_with_cam,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3), gr.Image(type="numpy")],
    title="E-Waste Image Classifier with Grad-CAM"
)

interface.launch(share=True)

