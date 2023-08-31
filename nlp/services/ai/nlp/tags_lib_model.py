import os
import nltk
from os.path import join, dirname
from dotenv import load_dotenv
from underthesea import ner
# Stop word
stop_word = ["Việt Nam"]

def group_entities(entities):
	grouped_entities = []
	current_entity = None

	for entity in entities:
		if entity['entity'].startswith('B-'):
			if current_entity:
				grouped_entities.append(current_entity)
			current_entity = {'type': entity['entity'][2:], 'start': entity['start'], 'end': entity['end'], 'name': entity['word']}
		elif entity['entity'].startswith('I-'):
			if current_entity and current_entity['type'] == entity['entity'][2:]:
				current_entity['end'] = entity['end']
				current_entity['name'] += " "+ entity['word']
			else:
				if current_entity:
					grouped_entities.append(current_entity)
				current_entity = None

	if current_entity:
		grouped_entities.append(current_entity)

	return grouped_entities
def generate_lib_tag(text):
	''' Public method generate English Tags with Libraries '''
	text = __remove_special_character(text)
	entities = ner(text, deep=True)
	grouped_entities = group_entities(entities)
	return grouped_entities


def __remove_special_character(text):
	''' Remove special characters '''
	if (">>" in text):
		text = text.split('>>')[1]
	text = text.replace("(", " ")
	text = text.replace(")", " ")
	text = text.replace("/", " ")
	text = text.replace('\\', " ")
	text = text.replace(":", " ")
	text = text.replace(",", " ")
	text = text.replace("'"," ")
	text = text.replace('"'," ")
	return text