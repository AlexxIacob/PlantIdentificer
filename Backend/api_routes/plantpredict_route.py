from flask import Flask, request, jsonify
from flask import Blueprint
from plant_classification_model.model import predict_file


plantpredict_bp = Blueprint('plantpredict', __name__)

@plantpredict_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        result = predict_file(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


