from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import asyncio
from functools import wraps

# Helper function to run async code in Flask
def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "InputImage.jpg"
        # Load the Keras model
        self.classifier = tf.keras.models.load_model('plant_disease_classifier_model.keras')
        # Initialize Groq LLM
        self.chat = ChatGroq(
            temperature=0.3,
            # groq_api_key=os.getenv('GROQ_API_KEY'),
            groq_api_key = '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',
            model_name="llama3-8b-8192"
        )
        self.setup_prompt()
        # Define confidence threshold
        self.confidence_threshold = 0.7  # 50% confidence threshold

    def setup_prompt(self):
        """Set up a single prompt template for plant analysis"""
        system_context = """You are a plant pathologist familiar with diseases affecting tomatoes, peppers, and potatoes. 
        You provide brief, practical information. You know about these specific conditions:
        - Bell Pepper Bacterial Spot
        - Potato Early Blight
        - Potato Late Blight
        - Tomato Bacterial Spot
        - Tomato Early Blight
        - Tomato Late Blight
        - Tomato Leaf Mold
        - Tomato Septoria Leaf Spot
        - Tomato Spider Mites
        - Tomato Target Spot
        - Tomato Yellow Leaf Curl Virus
        - Tomato Mosaic Virus
        And you can identify healthy plants."""

        template = """The AI system has detected {condition} in the uploaded image.

        If this is a diseased plant: Provide a single paragraph describing the disease's main symptoms and the most important treatment steps.
        
        If this is a healthy plant: Simply provide a brief confirmation of the plant's health and 1-2 basic care tips."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_context),
            ("human", template)
        ])

    async def get_analysis(self, condition: str) -> str:
        """Get a single analysis response from LLM"""
        try:
            analysis_chain = self.prompt | self.chat
            response = await analysis_chain.ainvoke({"condition": condition})
            return response.content
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return f"Error analyzing the plant condition: {str(e)}"

# Your existing class_name_map remains the same
class_name_map = {
    'Pepper__bell___Bacterial_spot': 'Bell Pepper Bacterial Spot',
    'Pepper__bell___healthy': 'Healthy Bell Pepper',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Healthy Potato',
    'Tomato_Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato_Early_blight': 'Tomato Early Blight',
    'Tomato_Late_blight': 'Tomato Late Blight',
    'Tomato_Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato Spider Mites (Two-Spotted)',
    'Tomato__Target_Spot': 'Tomato Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato_healthy': 'Healthy Tomato'
}

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
@async_route
async def predictRoute():
    try:
        clApp = ClientApp()
        # Decode the incoming image
        image = request.json['image']
        decodeImage(image, clApp.filename)

        # Preprocess the image
        img = load_img(clApp.filename, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict using the model
        predictions = clApp.classifier.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        max_confidence = float(predictions[0][predicted_class])
        
        # Check if confidence is below threshold
        if max_confidence < clApp.confidence_threshold:
            result = {
                "prediction": "Unknown Plant",
                "explanation": "This doesn't appear to be an image of a potato, pepper, or tomato leaf. The model's confidence is too low for a reliable diagnosis. Please ensure you're uploading a clear image of a potato, pepper, or tomato plant leaf.",
                "confidence": f'{max_confidence*100:.2f}%'
            }
        else:
            # Process as normal for high confidence predictions
            predicted_label = list(class_name_map.values())[predicted_class]
            
            # Get analysis from LLM
            analysis = await clApp.get_analysis(predicted_label)
            
            # Format confidence score
            confidence = f'{max_confidence*100:.2f}%'
            
            # Prepare the result
            result = {
                "prediction": f'{predicted_label} ({confidence})',
                "explanation": analysis,
                "confidence": confidence
            }
        
        print('Prediction Result:', result)
        return jsonify(result)

    except Exception as e:
        print(f"Error in prediction route: {str(e)}")
        return jsonify({
            "error": "An error occurred during prediction",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
