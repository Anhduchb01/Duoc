from .services.tag_service import process_tag_lib,predict_tag_lib, process_tag_ai , run_process_training_tag,ckeck_database_tag_service,evaluate_tag_ai,predict_tag_ai,extract_data_for_tag
from .services.classification_service import run_process_classification,run_predict_classification,run_process_training_classification,ckeck_database_classification_service
from .services.ai_modelService.modelService import clone_model_ai,remove_model_ai,clone_model_classification_ai,remove_model_classification_ai
from .repositories.article_repository import extract_article_verify
from flask import Flask, jsonify, request,Blueprint
from flask_cors import CORS
import json
import numpy as np
from json import JSONEncoder

nlp = Blueprint('nlp', __name__)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@nlp.route('/process-tag/model-lib', methods=['GET', 'POST'])
def execute_lib():
    ''' Execute Libaries to generate tag '''
    page = request.args.get('page')
    ckeck_database_tag_service(page)
    str = process_tag_lib()
    return jsonify({"message": str})

# Run Model Ai
@nlp.route('/process-tag/model-ai', methods=['GET', 'POST'])
def execute_ai():
    ''' Execute AI model to generate tag '''
    page = request.args.get('page')
    print(page)
    ckeck_database_tag_service(page)
    id = request.args.get('id') 
    time = request.args.get('time')
    str = process_tag_ai(id,time)
    return jsonify({"message": str})

# Predict Model Library for Play Ground
@nlp.route('/predict-tag/model-lib', methods=['GET', 'POST'])
def predict_lib():
    ''' Execute Libaries to generate tag '''
    page = request.form.get('page')
    ckeck_database_tag_service(page)
    lang = request.form.get('language') 
    
    text = str(request.form.get('text'))
    result = predict_tag_lib(text)
    return jsonify({"message": result})

# Predict Model AI for Play Ground
@nlp.route('/predict-tag/model-ai', methods=['GET', 'POST'])
def predict_ai():
    ''' Execute AI model to generate tag '''
    page = request.form.get('page')
    ckeck_database_tag_service(page)
    lang = request.form.get('language') 
    id = request.form.get('id') 
    time = request.form.get('time')
    text = request.form.get('text')
    result = predict_tag_ai(lang, id,time,text)

    return json.dumps({'message': result}, cls=NpEncoder)

# Run evaluate tag Ai
@nlp.route('/evaluate-tag/model-ai', methods=['GET', 'POST'])
def evaluate_ai():
    ''' Evaluate AI model to generate tag '''
    page = request.args.get('page')
    ckeck_database_tag_service(page)
    lang = request.args.get('language') 
    id = request.args.get('id') 
    
    str = evaluate_tag_ai(lang, id)
    return jsonify({"message": str})

# Extract data for tag
@nlp.route('/extract-data-for-tag', methods=['GET', 'POST'])
def extract_data_tag():
    str = extract_data_for_tag()
    return jsonify({"data":str})
# Extract data for sentiment
@nlp.route('/extract-data-for-sentiment', methods=['GET', 'POST'])
def extract_data_sentiment():
    str = extract_article_verify()
    return jsonify({"data":str})

# Run Classification AI
@nlp.route('/process-classification/model-ai', methods=['GET', 'POST'])
def process_classification():
    ''' Execute AI model to generate tag '''
    page = request.args.get('page')
    ckeck_database_classification_service()
    id = request.args.get('id')
    time = request.args.get('time')
    str = run_process_classification( id,time)
    return jsonify({"message": str})

# Predict Classification AI for Play Ground
# Run Classification AI
@nlp.route('/predict-classification/model-ai', methods=['GET', 'POST'])
def predict_classification():
    ''' Execute AI model to generate tag '''
    page = request.form.get('page')
    ckeck_database_classification_service(page)
    lang = request.form.get('language') 
    id = request.form.get('id')
    time = request.form.get('time')
    text = request.form.get('text')
    print(id)
    result = run_predict_classification( id,time,text)
    return jsonify({"message": result})

@nlp.route('/clone-model-ai', methods=['GET', 'POST'])
def clone_model():
    id = request.args.get('id') 
    idParent = request.args.get('idparent') 
    str = clone_model_ai(id,idParent)
    return jsonify({"message": str})

@nlp.route('/remove-model', methods=['GET', 'POST'])
def remove_model():
    id = request.args.get('id') 
    str = remove_model_ai(id)
    return jsonify({"message": str})

# Run Tag AI
@nlp.route('/training-tag/model-ai', methods=['GET', 'POST'])
def process_training_tag():
    ''' Execute AI model to generate tag '''
    print('start training model tag')
    page = request.args.get('page')
    ckeck_database_tag_service(page)
    id = request.args.get('id') 
    str = run_process_training_tag(id)
    print('finish training model tag')
    return jsonify({"message": str})


# Run Classification AI
@nlp.route('/training-classification/model-ai', methods=['GET', 'POST'])
def process_training_classification():
    ''' Execute AI model to generate tag '''
    print('start training')
    page = request.args.get('page')
    ckeck_database_classification_service()
    print('page',page)
    id = request.args.get('id') 
    print('id',id)
    str = run_process_training_classification(id)
    return jsonify({"message": str})

@nlp.route('/process-classification/clone-model-ai', methods=['GET', 'POST'])
def clone_model_classification():
    id = request.args.get('id') 
    idParent = request.args.get('idparent') 
    str = clone_model_classification_ai(id,idParent)
    return jsonify({"message": str})

@nlp.route('/process-classification/remove-model', methods=['GET', 'POST'])
def remove_model_classification():
    id = request.args.get('id') 
    str = remove_model_classification_ai(id)
    return jsonify({"message": str})


