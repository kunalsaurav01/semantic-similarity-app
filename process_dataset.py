# --- 1. Import Necessary Libraries ---
# pandas is a powerful library for data manipulation, perfect for handling CSV files.
import pandas as pd
# SentenceTransformer and util are used for the core similarity calculation.
from sentence_transformers import SentenceTransformer, util

def analyze_dataset(input_csv_path):
    """
    This function orchestrates the entire process of reading the dataset,
    calculating similarity scores, and saving the results.

    Args:
        input_csv_path (str): The file path for the input CSV.
    """
    # --- a. Load the Model ---
    # We load the same model as in the API for consistency.
    print("Loading the Sentence-BERT model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # --- b. Read the Dataset ---
    # We use a try-except block to handle potential errors, like the file not being found.
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded '{input_csv_path}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- c. Prepare Data for Batch Processing ---
    # The model works most efficiently when given a list of sentences (a "batch").
    # We convert the 'text1' and 'text2' columns from the DataFrame into Python lists.
    # We also convert any non-string data to strings to prevent errors.
    sentences1 = df['text1'].astype(str).tolist()
    sentences2 = df['text2'].astype(str).tolist()

    # --- d. Encode Texts in Batches ---
    print("\nEncoding all texts into vector embeddings. This might take some time...")
    # The model processes the entire list of sentences at once, which is much faster
    # than looping through each row one by one.
    embeddings1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)
    print("Encoding complete.")

    # --- e. Calculate Similarity Scores ---
    # This computes the cosine similarity for each corresponding pair of embeddings
    # (e.g., embedding1[0] vs embedding2[0], embedding1[1] vs embedding2[1], etc.).
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # --- f. Extract and Store Results ---
    similarity_scores = []
    # We loop through the number of sentences to extract each score.
    for i in range(len(sentences1)):
        # The score for the i-th pair is on the diagonal of the results matrix.
        score = cosine_scores[i][i].item()
        # We scale the score to the [0, 1] range as required.
        scaled_score = (score + 1) / 2
        similarity_scores.append(round(scaled_score, 4))

    # --- g. Save the Results ---
    # We add the list of scores as a new column to our original DataFrame.
    df['similarity_score'] = similarity_scores

    # Define the output filename.
    output_filename = 'DataNeuron_Similarity_Results.csv'
    # Save the updated DataFrame to a new CSV file.
    # index=False prevents pandas from writing the DataFrame index as a column.
    df.to_csv(output_filename, index=False)

    print("\n--- Analysis Complete ---")
    print("Top 5 rows of the results:")
    print(df.head())
    print(f"\nFull results have been saved to '{output_filename}'")


# This block ensures the code runs only when the script is executed directly.
if __name__ == '__main__':
    # Set the path to your dataset file.
    dataset_file_path = 'DataNeuron_Text_Similarity.csv'
    analyze_dataset(dataset_file_path)

