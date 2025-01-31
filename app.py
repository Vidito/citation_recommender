import os
import pickle
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024  # Limit input size to 10 KB
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load journal data (i already did to use embeddings)
journals = pd.read_json("journals.json")

# Prepare journal descriptions for embeddings (i already did to use embeddings)
# journals["description"] = (
#     journals["scope_aims"]
#     + " "
#     + journals["keywords"].apply(lambda x: " ".join(map(str, x)))
# )

# Check if precomputed embeddings exist
EMBEDDINGS_FILE = "journals_embeddings.npy"
journals_embeddings = np.load(EMBEDDINGS_FILE)

# I can also use some checks
# if os.path.exists(EMBEDDINGS_FILE):
#     journals_embeddings = np.load(EMBEDDINGS_FILE)
# else:
#     journals_embeddings = model.encode(journals["description"], convert_to_numpy=True)
#     np.save(EMBEDDINGS_FILE, journals_embeddings)


def preprocess_text(text):
    doc = nlp(text.lower())  # Convert to lowercase
    cleaned_text = " ".join(
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct
    )
    return cleaned_text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend_journals():
    data = request.get_json()
    abstract = data.get("abstract", "")
    abstract = preprocess_text(abstract)

    # Validate input structure
    if not isinstance(data, dict) or "abstract" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    # Check for empty or excessively long input
    if not abstract:
        return jsonify({"error": "Abstract is required"}), 400
    if len(abstract.split()) > 400:  # Limit to 200 words
        return jsonify({"error": "Abstract too long"}), 400

    # Compute embedding for user input
    user_embedding = model.encode([abstract], convert_to_numpy=True)

    # Compute similarity
    similarities = cosine_similarity(user_embedding, journals_embeddings)[0]

    # Get top 5 most similar journals
    top_indices = similarities.argsort()[-7:][::-1]  # Get top 5 indices, sorted

    recommendations = []
    for i in top_indices:
        recommendations.append(
            {
                "name": journals.loc[i, "name"],
                "scope": " ".join(journals.loc[i, "scope_aims"].split()[:20]) + "...",
                "link": journals.loc[i, "link"],
                "impact_factor": journals.loc[i, "impact_factor"],
            }
        )

    return jsonify(recommendations)


if __name__ == "__main__":
    app.run(debug=True)
