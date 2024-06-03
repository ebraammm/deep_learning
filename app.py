from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import nltk
import google.generativeai as genai
import chromadb
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

app = Flask(__name__)

# Root route
@app.route('/')
def home():
    return "Welcome to the PDF Text Extraction and Query API!"

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Improved text splitting
def split_text(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Step 3: Generate embeddings with Google Gemini API using `models/text-embedding-004`
API_KEY = 'AIzaSyDleZ4xVF9dCT7aw95WBeDpfHwktn4LUQ0'  # Replace with your actual API key
genai.configure(api_key=API_KEY)

class GeminiEmbeddingFunction:
    def __call__(self, input):
        model = 'models/text-embedding-004'
        response = genai.embed_content(model=model, content=input, task_type="retrieval_document")
        if 'embedding' in response:
            return response['embedding']
        else:
            raise KeyError(f"'embedding' key not found in response: {response}")

# Step 4: Store embeddings in ChromaDB
def create_chroma_db(documents, name):
    chroma_client = chromadb.Client()

    # Check if the collection exists and delete it
    try:
        chroma_client.delete_collection(name=name)
    except KeyError:
        pass  # Collection does not exist

    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(documents):
        db.add(
            documents=[d],
            ids=[str(i)]
        )
    return db

db = None

@app.route('/initialize', methods=['POST'])
def initialize_db():
    global db
    pdf_path = request.json.get('pdf_path')
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(pdf_text)
    db = create_chroma_db(text_chunks, "egypt_places_chromadb")
    return jsonify({"message": "Database initialized with PDF content."})

# Step 5: Query the Database
def get_relevant_passage(query, db):
    result = db.query(query_texts=[query], n_results=1)
    passage = result['documents'][0][0]
    return passage

def generate_response(query, passage):
    prompt = f"Query: {query}\nPassage: {passage}\nResponse:"
    response = genai.generate_text(prompt)
    return response['text']

@app.route('/query', methods=['POST'])
def query_db():
    global db
    query = request.json.get('query')
    passage = get_relevant_passage(query, db)
    response = generate_response(query, passage)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
