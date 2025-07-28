from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.rag_core import rag_system
from src.pdf_processor import extract_text_from_pdf, chunk_text
import uuid

app = Flask(__name__)
CORS(app)

session_storage = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    pdf_file = request.files['pdf']
    
    try:
        raw_text = extract_text_from_pdf(pdf_file)
        text_chunks = chunk_text(raw_text)
        index = rag_system.build_index_from_chunks(text_chunks)
        
        session_id = str(uuid.uuid4())
        session_storage[session_id] = {
            'index': index,
            'chunks': text_chunks
        }
        
        return jsonify({'session_id': session_id, 'message': 'PDF processed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id')

    if not all([question, session_id]):
        return jsonify({'error': 'Missing question or session_id'}), 400
    
    session_data = session_storage.get(session_id)
    if not session_data:
        return jsonify({'error': 'Invalid session. Please upload the PDF again.'}), 404
        
    # This is the corrected function call.
    # It passes the query, the FAISS index, and the text chunks in the correct order.
    answer = rag_system.get_answer(
        question, 
        session_data['index'], 
        session_data['chunks']
    )
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
