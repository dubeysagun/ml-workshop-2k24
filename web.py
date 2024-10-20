import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib  # To save and load the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Paths to save the models
TEXT_MODEL_PATH = 'text_classifier.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
IMAGE_MODEL_PATH = 'image_classifier.h5'

# Load and preprocess text data
def load_and_preprocess_text_data():
    # Load text data with correct encoding and handle bad characters
    fake_news_train = pd.read_csv('fake_news_train.csv', encoding='ISO-8859-1', on_bad_lines='skip')
    fake_news_test = pd.read_csv('fake_news_test.csv', encoding='ISO-8859-1', on_bad_lines='skip')
    real_news_train = pd.read_csv('train_real_news.csv', encoding='ISO-8859-1', on_bad_lines='skip')
    real_news_test = pd.read_csv('test_real_news.csv', encoding='ISO-8859-1', on_bad_lines='skip')

    # Combine train and test data
    train_data = pd.concat([fake_news_train, real_news_train])
    test_data = pd.concat([fake_news_test, real_news_test])

    # Add labels (1 for real, 0 for fake)
    train_data['label'] = [0] * len(fake_news_train) + [1] * len(real_news_train)
    test_data['label'] = [0] * len(fake_news_test) + [1] * len(real_news_test)

    # Remove unwanted characters from the 'News' column
    train_data['News'] = train_data['News'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    test_data['News'] = test_data['News'].str.replace(r'[^\x00-\x7F]+', '', regex=True)

    # Handle missing values in 'News' column by filling with an empty string
    train_data['News'] = train_data['News'].fillna('')  # Replace NaN with empty string
    test_data['News'] = test_data['News'].fillna('')  # Replace NaN with empty string

    return train_data, test_data

# Train or load the text classifier
def train_text_classifier(train_data):
    # Check if model exists
    try:
        rf_model = joblib.load(TEXT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        st.write("Loaded pre-trained text classifier.")
    except FileNotFoundError:
        st.write("Training new text classifier...")

        # More aggressive text preprocessing
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_data['News'].str.lower())  # Convert text to lowercase
        y_train = train_data['label']

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
        rf_model.fit(X_train, y_train)

        # Validation accuracy
        val_preds = rf_model.predict(X_val)
        accuracy = accuracy_score(y_val, val_preds)
        st.write(f"Text Classifier Validation Accuracy: {accuracy * 100:.2f}%")

        # Save the trained model and vectorizer
        joblib.dump(rf_model, TEXT_MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        st.write("Text classifier model saved.")

    return rf_model, vectorizer

# Train or load the image classifier
def train_image_classifier():
    # Check if model exists
    try:
        model = load_model(IMAGE_MODEL_PATH)
        st.write("Loaded pre-trained image classifier.")
    except OSError:
        st.write("Training new image classifier...")

        # Image Data Generator for Augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Train and Test directories
        train_dir = 'images/train/'
        test_dir = 'images/test/'

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),  # Use 224x224 size for VGG16
            batch_size=32,
            class_mode='binary'
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        # Load the VGG16 model pre-trained on ImageNet
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers on top of the base model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
        predictions = Dense(1, activation='sigmoid')(x)

        # Define the new model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Training the model
        model.fit(train_generator, epochs=10, validation_data=test_generator)

        # Save the trained model
        model.save(IMAGE_MODEL_PATH)
        st.write("Image classifier model saved.")

    return model

# Predict text function
def predict_text(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text.lower()])  # Apply the same transformations
    prediction = model.predict(text_vectorized)
    return "Real" if prediction == 1 else "Fake"

# Predict image function
def predict_image(img, model):
    img_resized = cv2.resize(img, (224, 224))  # Resize to match the model input size
    img_array = np.expand_dims(img_resized, axis=0) / 255.0  # Normalize and add batch dimension
    prediction = model.predict(img_array)
    return "Real" if prediction > 0.5 else "Fake"

# Main Streamlit App
def main():
    st.title("Fake News Detection System")
    st.write("Enter news text or upload an image to predict if it's fake or real")

    # Load and train models (if not already saved)
    train_data, test_data = load_and_preprocess_text_data()
    text_model, vectorizer = train_text_classifier(train_data)
    image_model = train_image_classifier()

    # User input (text or image)
    option = st.selectbox("Choose input type", ("Text", "Image"))

    if option == "Text":
        user_input = st.text_area("Enter the news text")
        if st.button("Predict"):
            result = predict_text(user_input, text_model, vectorizer)
            st.write(f"The news is: {result}")
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR")
            if st.button("Predict"):
                result = predict_image(img, image_model)
                st.write(f"The news is: {result}")

if __name__ == '__main__':
    main()
