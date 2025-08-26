# --- 1. Import Necessary Libraries ---
# Flask is used to create the web application and handle API requests.
# request helps access incoming data (like JSON).
# jsonify is used to format our Python dictionary response into a proper JSON format.
from flask import Flask, request, jsonify

# SentenceTransformer is the main library for using the SBERT models.
# util provides helper functions, including the one for cosine similarity.
from sentence_transformers import SentenceTransformer, util

# --- 2. Initialize the Flask Application ---
# This creates an instance of the Flask web server.
app = Flask(__name__)

# --- 3. Load the Machine Learning Model ---
# This is a crucial step. We load the pre-trained Sentence-BERT model.
# 'all-MiniLM-L6-v2' is a powerful and compact model, great for semantic similarity.
# The model is loaded only ONCE when the application starts. This is efficient
# because it avoids reloading the model for every single API request.
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# --- 4. Define API Routes (Endpoints) ---

# This defines the root URL (e.g., http://your-app.com/).
# It's a simple health check to confirm the server is running.
@app.route('/')
def home():
    return "<h1>Semantic Similarity API</h1><p>The server is running. Use the /predict endpoint to get similarity scores.</p>"

# This defines the main endpoint for our service.
# It's accessible at '/predict' and only accepts POST requests.
@app.route('/predict', methods=['POST'])
def predict():
    """
    This function is executed when a POST request is made to /predict.
    It expects a JSON body with two keys: "text1" and "text2".
    It returns a JSON response with the calculated "similarity_score".
    """
    # --- a. Get and Validate Input Data ---
    # Get the JSON data sent in the request body.
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON received.'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to parse JSON: {e}'}), 400

    # Check if the required keys 'text1' and 'text2' are in the JSON.
    if 'text1' not in data or 'text2' not in data:
        return jsonify({'error': 'Request body must contain both "text1" and "text2" keys.'}), 400

    # Extract the text paragraphs from the data.
    text1 = data['text1']
    text2 = data['text2']

    # --- b. Perform the Model Inference (Part A Logic) ---
    # The model's 'encode' method converts the text strings into numerical vectors (embeddings).
    # These embeddings capture the semantic meaning of the text.
    # 'convert_to_tensor=True' prepares the embeddings for similarity calculation.
    print("Encoding texts into embeddings...")
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Calculate the cosine similarity between the two embeddings.
    # This gives a score indicating how similar the vectors (and thus the texts) are.
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    
    # The result is a tensor; .item() extracts the raw numerical value.
    similarity_score = cosine_scores.item()

    # The problem requires a score between 0 and 1.
    # Cosine similarity is technically between -1 and 1. We scale it to the [0, 1] range
    # using the formula: (score + 1) / 2. This ensures the output always meets the requirement.
    scaled_score = (similarity_score + 1) / 2

    # --- c. Format and Return the Response ---
    # Create the response dictionary in the specified format.
    # We round the score for a cleaner output.
    response = {'similarity_score': round(scaled_score, 4)}

    # Convert the dictionary to a JSON string and send it back to the client.
    print(f"Prediction successful: {response}")
    return jsonify(response)

# --- 5. Run the Application ---
# This block checks if the script is being run directly (not imported).
# If so, it starts the Flask development server.
# host='0.0.0.0' makes the server accessible from other devices on the network.
# Heroku will use a production server (Gunicorn), so this part is mainly for local testing.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)