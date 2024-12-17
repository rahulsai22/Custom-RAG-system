
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Load environment variables
load_dotenv()

# Initialize OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Initialize LlamaIndex components
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)

retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.50)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor]
)

# Initialize Flask app
app = Flask(__name__)

# Route for serving index.html
@app.route('/')
def home():
    """Serve the HTML page."""
    return render_template('index.html')

# API endpoint for querying
@app.route('/query', methods=['POST'])
def query():
    """Handle query requests."""
    try:
        # Parse JSON request
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({"error": "Please provide a 'question' in the request body."}), 400

        # Process query using LlamaIndex
        response = query_engine.query(question)
        return jsonify({"response": str(response)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("127.0.0.1", 5000, app, use_reloader=False, use_debugger=True)

