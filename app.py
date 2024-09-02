from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load the model from Sentence Transformers
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load your DataFrame
df = pd.read_csv('anime.csv')

# Create a new column with combined information and ensure all values are strings
df['description'] = df['name'] + ' ' + df['genre'] + ' ' + df['type'] + ' episodes: ' + df['episodes'].astype(str)
df['description'] = df['description'].fillna('')  # Replace NaN with empty string if any
df['description'] = df['description'].astype(str)  # Convert all values to string

# Compute embeddings for all anime descriptions
anime_embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True)

def get_recommendations(query, anime_embeddings, df):
    # Encode the query
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Move the embeddings to the CPU before calculating cosine similarity
    query_embedding = query_embedding.cpu()
    anime_embeddings = anime_embeddings.cpu()
    
    # Compute cosine similarities between the query and anime embeddings
    similarities = cosine_similarity(query_embedding.numpy(), anime_embeddings.numpy())
    
    # Get top 20 similar items before filtering duplicates
    top_indices = similarities[0].argsort()[-20:][::-1]
    recommendations = df.iloc[top_indices]
    
    # Drop duplicates based on the 'name' column
    recommendations = recommendations.drop_duplicates(subset=['name'])
    
    # Limit to the top 5 recommendations after dropping duplicates
    recommendations = recommendations.head(5)
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        query = request.form['query']
        recommendations = get_recommendations(query, anime_embeddings, df)
        recommendations = recommendations[['name']].to_dict(orient='records')
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
