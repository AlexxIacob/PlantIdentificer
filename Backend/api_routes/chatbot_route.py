from flask import request, jsonify
from flask import Blueprint
from services.chatbot_service import get_model_response
chatbot_bp = Blueprint('chatbot', __name__)


@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('message', '')

    if not question:
        return jsonify({"error": "Întrebarea este obligatorie"}), 400

    answer = get_model_response(question)
    return jsonify({"answer": answer})

    return jsonify({'answer': response_text}), 200
