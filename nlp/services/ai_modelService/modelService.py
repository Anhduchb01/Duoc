import os
from pathlib import Path
import shutil
from transformers import pipeline, AutoModelForTokenClassification,AutoTokenizer ,AutoModelForSequenceClassification

current_path = Path(__file__).parent.parent.parent.joinpath('ai_model')

def clone_model_ai(id, idParent):
    ''' Clone NER model '''
    if idParent == 'default':
        destination = shutil.copytree(current_path.joinpath("ner-default"), current_path.joinpath(id))
        return {"data": "finish"}
    else:
        destination = shutil.copytree(current_path.joinpath(idParent), current_path.joinpath(id))
        return {"data": "finish"}

def remove_model_ai(id):
    ''' Remove NER model '''
    remove = shutil.rmtree(current_path.joinpath(id))
    return {"data": "finish"}


def clone_model_classification_ai(id, idParent):
    ''' Clone Classification model'''
    idParent = str(idParent)
    print(idParent)
    if idParent == 'default':
        destination = shutil.copytree(current_path.joinpath('sentiment-default'), current_path.joinpath(id))
        return {"data": "finish"}
    else:
     
        destination = shutil.copytree(current_path.joinpath(idParent), current_path.joinpath(id))
        return {"data": "finish"}

def remove_model_classification_ai(id):
    ''' Remove Classification model '''
    remove = shutil.rmtree(current_path.joinpath(id))
    return {"data": "finish"}