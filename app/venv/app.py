from flask import Flask, request, jsonify, render_template
import re
import chromadb
import google.generativeai as genai
import os
from PyPDF2 import PdfReader

app = Flask(__name__)

# Configure Google Generative AI
API_KEY = "AIzaSyDleZ4xVF9dCT7aw95WBeDpfHwktn4LUQ0"
genai.configure(api_key=API_KEY)

# Path to the PDF document
PDF_PATH = "C:\\Users\\MN\\Desktop\\deep_learning\\dataset\\best 55 places.pdf"


# Define functions from the notebook
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = "\n".join(page.extract_text() for page in reader.pages)
    return text

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input):
        model = 'models/text-embedding-004'
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

def create_chroma_db(documents, name):
    chroma_client = chromadb.Client()
    existing_collections = chroma_client.list_collections()
    existing_collection_names = [collection.name for collection in existing_collections]

    if name in existing_collection_names:
        chroma_client.delete_collection(name=name)

    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for i, doc in enumerate(documents):
        db.add(documents=[doc], ids=[str(i)])
    return db

def get_relevant_passage(query, db):
    passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
    return passage

def make_prompt(query, relevant_passage):
    prompt = (
        f"You are a tour guide Bot, a travel assistant specialized in Egyptian destinations. "
        f"Answer questions using the reference passage below. "
        f"Provide detailed information  about specific places, including name, description, location and timings with images. "
        f"Maintain a friendly, knowledgeable tone. "
        f" you can recommend places to visit after the visits, restaurants and cafes in the same place"
        f"QUESTION: '{query}' "
        f"ANSWER:"
    )
    return prompt

def generate_response_with_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.0-pro')
    answer = model.generate_content(prompt)
    return answer.text

def chat_with_pdf(query, db):
    response = "No document."
    document_text = get_relevant_passage(query, db)
    if document_text:
        prompt = make_prompt(query, document_text)
        response = generate_response_with_gemini(prompt)
    return response

# Load and process the PDF once at startup
text = extract_text_from_pdf(PDF_PATH)
cleaned_text = preprocess_text(text)
documents = [cleaned_text]
collection = create_chroma_db(documents, "egypt_travel")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"})
    response = chat_with_pdf(user_query, collection)
    return jsonify({"response": response})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
