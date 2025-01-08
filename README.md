# Plant Disease Detection System

## Overview
This application uses deep learning and AI to detect diseases in plants (specifically tomatoes, peppers, and potatoes) from images. It combines a TensorFlow-based image classifier with the Groq LLM to provide detailed analysis and treatment recommendations.

## Features
- Image-based disease detection for multiple plant types
- Supported plants: Tomatoes, Peppers, Potatoes
- Disease classification with confidence scores
- AI-powered analysis of plant conditions
- Treatment recommendations for detected diseases
- Support for healthy plant identification

## Diseases Detected
### Tomato
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus

### Pepper
- Bacterial Spot

### Potato
- Early Blight
- Late Blight

## Technical Stack
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras
- **AI Language Model**: Groq LLM
- **Image Processing**: TensorFlow Preprocessing
- **API**: REST API with CORS support
- **Data Format**: JSON