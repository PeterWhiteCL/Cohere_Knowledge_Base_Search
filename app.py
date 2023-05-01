from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd 
import cohere
from io import StringIO
import json
from typing import Any, List
from io import StringIO
from numpy.linalg import norm
from PyPDF2 import PdfReader
import os

api_key = os.environ["API_KEY"]
CHUNK_SIZE = 512
OUTPUT_BASE_DIR = "./"
co = cohere.Client(api_key)


def process_text_input(text: str, run_id: str = None):
    text = StringIO(text).read()
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    df = pd.DataFrame.from_dict({'text': chunks})

    return df

def get_embedding(query):
    embed = co.embed(texts=query,
                  model="large").embeddings
    return embed

def select_prompts(list_of_texts, sorted_indices):
    return np.take_along_axis(np.array(list_of_texts)[:, None], sorted_indices, axis=0)

def top_n_neighbours_indices(prompt_embedding: np.ndarray, storage_embeddings: np.ndarray, n: int = 5):
    if isinstance(storage_embeddings, list):
        storage_embeddings = np.array(storage_embeddings)
    if isinstance(prompt_embedding, list):
        prompt_embedding = np.array(prompt_embedding)        
    similarity_matrix = prompt_embedding @ storage_embeddings.T / np.outer(norm(prompt_embedding, axis=-1), norm(storage_embeddings, axis=-1))
    num_neighbours = min(similarity_matrix.shape[1], n)
    indices = np.argsort(similarity_matrix, axis=-1)[:, -num_neighbours:]

    return indices

def get_closest_passage(prompt_embedding, storage_embeddings, storage_df) -> List:
    assert prompt_embedding.shape[0] == 1 
    if isinstance(prompt_embedding, list):
        prompt_embedding = np.array(prompt_embedding)
    indices = top_n_neighbours_indices(prompt_embedding, storage_embeddings, n=5)
    similar_passages = select_prompts(storage_df.text.values, indices)

    return similar_passages[0]



##Retrieve text from text file instead of reading line-by-line from PDF
file = open("kbtext.json")
text = json.load(file)
file.close()



# This section deals with Embedding the Knowledge Base  
# Get reference dataframe from text
# reader = PdfReader('i1040gi.pdf')
# text = ''

# for page in reader.pages:
#     text = text + page.extract_text()

# with open("kbtext.json", "w") as f:
#     json.dump(text,f)



df = None
df = process_text_input(text)

# embeddings = get_embedding(list(df.text.values))

jsonFile = open("kbembeddings.json")
embeddings = np.array(json.load(jsonFile))
jsonFile.close()


# with open("kbembeddings.json", "w") as f:
#     json.dump(embeddings, f)

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    # Get the search query from the user
    query = request.args.get('query')

    # Convert the query into an embedding
    query_embedding = get_embedding([query])
    
    closest_passage = get_closest_passage(np.array(query_embedding), embeddings, df)

    prompt = '\n'.join(closest_passage) + '\n\n' + "Based on the passage above, answer the following question:" + '\n' + query + '\n'
    response = co.generate( 
        model='command-xlarge-nightly', 
        prompt=prompt,
        max_tokens=100, 
        temperature=0.6, 
        return_likelihoods='NONE')

    return jsonify(response.generations[0].text)



if __name__ == '__main__':
    app.run()

