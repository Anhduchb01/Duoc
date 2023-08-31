from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, logging
from pathlib import Path


# Current os path
current_path = Path(__file__).parent.parent.parent.parent.joinpath('ai_model')

# Logging setting
logging.set_verbosity_warning()
logging.set_verbosity_error()
def initialization_model_evaluate(id):
    model = AutoModelForTokenClassification.from_pretrained(current_path.joinpath(id))
    tokenizer = AutoTokenizer.from_pretrained(current_path.joinpath(id))
    ner = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner
def initialization_model(id):
    ''' Init model for using. Call first before call generate_keyword() '''
   
    if id == 'default':
        # BERT default model
        print(current_path)
        model = AutoModelForTokenClassification.from_pretrained(current_path.joinpath('ner-default'))
        tokenizer= AutoTokenizer.from_pretrained(current_path.joinpath('ner-default'))
        ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return ner
    else:
        # Saved model
        model = AutoModelForTokenClassification.from_pretrained(current_path.joinpath(id))
        return model


    
    

def generate_keyword_default(sequence, ner):
    ''' Using transformer to generate tag  '''
    sequence = __remove_special_character(sequence)
    
    output = ner(sequence)
    keywords = []
    for i in output:
        
        keyword = __remove_stop_word(i['word'])
        
        if (keyword):
            if i['entity_group'] =='PERSON':
                keywords.append({'name': keyword, 'type': "PER", 'score': i['score']})
            elif i['entity_group'] =='ORGANIZATION':
                keywords.append({'name': keyword, 'type': "ORG", 'score': i['score']})
            elif i['entity_group'] =='LOCATION':
                keywords.append({'name': keyword, 'type': "LOC", 'score': i['score']})
            else :
                keywords.append({'name': keyword, 'type': "MISC", 'score': i['score']})
    tags = []
    for keyword in keywords:
        if (keyword['name'] not in tags):
            tags.append(keyword)
    return tags
def generate_keyword(sequence,  tokenizer, model):

    ''' Using transformer to generate tag  '''
    text_split = sequence.split()
    encoding = tokenizer(text_split,truncation=True, is_split_into_words=True,padding='max_length',max_length=128,return_tensors="pt")
    outputs   = model(**encoding)
    predicted_labels = outputs.logits.argmax(dim=-1).squeeze().cpu().numpy()
    actual_word_labels = predicted_labels[1 : len(text_split) + 1]
    entities = []
    current_entity = None
    import torch.nn.functional as F
    probs = F.softmax(outputs.logits, dim=-1)
    actual_word_probs = probs[0][1: len(text_split) + 1].max(dim=-1).values.cpu().detach().numpy()
    id2label = {0: 'B-LOCATION',
            1: 'B-MISCELLANEOUS',
            2: 'B-ORGANIZATION',
            3: 'B-PERSON',
            4: 'I-LOCATION',
            5: 'I-MISCELLANEOUS',
            6: 'I-ORGANIZATION',
            7: 'I-PERSON',
            8: 'O'}
    for word, label_id, prob in zip(text_split, actual_word_labels, actual_word_probs):
        label = id2label[label_id]

        if label == "O":
            # The current word is outside of any entity, reset the current_entity
            current_entity = None
        else :
            label_type, entity_type = label.split("-")
            if label_type == "B":
                # The current word is the beginning of a new entity
                if entity_type =='PERSON':
                    label = 'PER'
                elif entity_type =='ORGANIZATION':
                    label = 'ORG'
                elif entity_type =='LOCATION':
                    label = 'LOC'
                else:
                    label = 'MISC'
                current_entity = {'name': word, 'type': label, 'score': round(prob, 4)}
                entities.append(current_entity)
            elif label_type == "I" and current_entity is not None:
                # The current word is inside an entity, add it to the current_entity
                current_entity['name'] += " " + word
                current_entity['score'] = round((prob+current_entity['score'])/2,4)
    return entities
    

def split_sentence(text):
    ''' Split big sentences to small sentences. Max sentences < 300 word '''
    text_process = []
    text = __remove_special_character(text)
    text_split_new_line = text.split('\n')
    text_split_by_dot = []
    for t in text_split_new_line:
        if (len(t.split()) > 100):
            text_split = t.split('.')
            text_split_by_dot += text_split
        else:
            text_split_by_dot.append(t)
    join_text = ''
    for t in text_split_by_dot:
        if (len(t.split(' ')) > 5):
            if ((len(join_text.split(' ')) + len(t.split(' '))) < 100):
                join_text += t
                
            else:
                text_process.append(join_text)
                join_text = ''
    if (join_text):
        text_process.append(join_text)
    return text_process
def split_sentence_create(text):
    ''' Split big sentences to small sentences. Max sentences < 300 word '''
    text = __remove_special_character1(text)
    text_process = text.split('.')
    return text_process

def __remove_special_character(text):
    ''' Remove special characters '''
    if (">>" in text):
        text = text.split('>>')[1]
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("/", " ")
    text = text.replace(":", " ")
    text = text.replace("'"," ")
    text = text.replace('"'," ")
    return text
import re
def __remove_special_character1(text):
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(",", " ")
    text = text.replace("/", " ")
    ext = text.replace("”", " ")
    ext = text.replace("“", " ")
    text = text.replace(":", " ")
    text = text.replace("'"," ")
    text = text.replace('"'," ")
    text = text.replace('’'," ")
    text = text.replace("'"," ")
    text = text.replace("+", " ")
    text = text.replace("–", " ")
    text = text.replace("%", " ")
    text = text.replace('\r\n'," ")
    text = text.replace(']'," ")
    text = text.replace('['," ")
    text = text.replace("-","")
    text = text.replace("\\ufeff", "")
    text = text.replace('\\'," ")
    text = text.strip()
    return text

def __remove_stop_word(tag):
    ''' Remove stop word '''
    if ',' in tag:
        return None
    if (not tag[-1].isalpha()) and (not tag[-1].isnumeric()):
        return None
    if tag.startswith('#'):
        return None
    
    if tag.endswith('Of'):
        return None
    if len(tag) < 3 :
        return None
    if  len(tag.split(" ")) > 10:
        return None
    txt1 = tag.replace("-","")
    txt1 = txt1.replace(" ","")
    if txt1.isnumeric():
        return None
    tag = tag.replace("and","")
    tag = tag.replace("for","")
    tag = tag.replace("of","")

    return tag
