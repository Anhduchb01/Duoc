from logging import error
from bson.objectid import ObjectId
from nltk.util import pr
from datasets import load_dataset, load_metric,Dataset,  DatasetDict
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments,AutoConfig,AutoModelForTokenClassification
import numpy as np
from pathlib import Path
import shutil
import time
from ..repositories.article_repository import ckeck_database_article_repository,get_unprocess_classification_article_ai, count_unprocess_classification_article_ai, update_article, update_status_model, find_model, get_article_verify
from ..services.ai.classification.classification_model import initialization_model, execute_model , get_split,execute_model_default

from pymongo import MongoClient
import datetime
import os
import math
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.environ.get('DB_URL')
client = MongoClient(DB_URL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
current_path = Path(__file__).parent.parent.joinpath('ai_model')
# client = MongoClient("mongodb://crawl02:crawl02123@localhost:27017/?authSource=FinSight")
historyClassification_collection = client["FinSight"]["historyclassifications"]
article_collection = client["FinSight"]["posts"]

def ckeck_database_classification_service():
	ckeck_database_article_repository()
	global historyClassification_collection
	global article_collection
	historyClassification_collection = client["FinSight"]["historyclassifications"]
	article_collection = client["FinSight"]["posts"]
def run_predict_classification(id, timeModel,text):
	
	
	classificator = initialization_model(id)
	print('create model ok')
	
	text_process = get_split(text)      
		
	if id == 'default':
		resultclassification = execute_model_default(text_process, classificator)
	else :
		resultclassification = execute_model(text_process, classificator)
	print('predict ok')
  
	
		
	return resultclassification

def run_process_classification(id, timeModel):
	''' Use Bert model to generate tag '''
	# Update Model status to processing
	timeExcute = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	timeExcute = str(timeExcute)
	if id == 'default' :
		update_status_model({"name": 'AI Sentiment Analysis Base'}, {"$set": {"status": 1}})
		# findModelBase = historyClassification_collection.find_one({"model_id": str(id)})
		# if findModelBase:
		#     print('model da ton tai')
		ArticleHasProcessed = historyClassification_collection.find_one({"model_id": str(id)})
		if ArticleHasProcessed:
			timeExcute = ArticleHasProcessed['time']

	else:
		update_status_model({"_id": ObjectId(id)}, {"$set": {"status": 1}})
	try:
		classificator = initialization_model(id)

		cursor_en =article_collection.find({'status':"0"},no_cursor_timeout=True).batch_size(20)
		for article in tqdm(cursor_en):
			ArticleHasProcessed = historyClassification_collection.find_one({"model_id": str(id),'article_id':ObjectId(article['_id'])})
			if ArticleHasProcessed:
				continue
			if article:
				# try:
					text = article["title"]
					if  text ==  None:
						continue
					text_process = get_split(text)
					
					if id == 'default':
						score_categories = execute_model_default(text_process, classificator)
					else :
						score_categories = execute_model(text_process, classificator)
					
					dict_score_categories = {}
					max_score = 0
					max_label = ''
					for label_dict in score_categories[0]:
						dict_score_categories[label_dict['label']] = label_dict['score']
						if label_dict['score'] > max_score:
							max_score = label_dict['score']
							max_label = label_dict['label']
							
					max_score_category = max_score

					# update_article({"_id": article["_id"]}, {"$set": {"category": max_score_category, "isClassification": True, 'classificationScore': score_categories}})
					historyClassification_collection.insert_one({'model_id':str(id),'time':timeExcute,'article_id':ObjectId(article["_id"]),'classificationScore':dict_score_categories,"category": max_label})

				# except Exception as e:
				# 	print("ERROR: " + str(e))
			else:
				break
		cursor_en.close()
		if id == 'default':
			update_status_model({"name": 'AI Sentiment Analysis Base'}, {"$set": {"status": 0}})
		else:
			update_status_model({"_id": ObjectId(id)}, {"$set": {"status": 0}})
		return {"data": "finish"}
	except Exception as e:
			print("ERROR: " + str(e))
			if id == 'default':
				model_collection.update_one({"name": "AI Sentiment Analysis Base"}, {"$set": {"status": 2}})
			
			else:
				model_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": 2}})

# Training Model
def run_process_training_classification(id):
	
		update_status_model({"_id": ObjectId(id)}, {"$set": {"status": 1}})
		try:
			arrayItem = get_article_verify()
			# df = pd.DataFrame(arrayItem)
			# df['text_split'] = '' 
			# for i in range(len(df)):
			# 	df['text_split'][i] = get_split(df['text_origin'][i])

			# df_data = pd.DataFrame(columns=['content','label'])
			# text = []
			# label = []
			# for i in range(len(df)):
			# 	for j in range(len(df['text_split'][i])):
			# 		text.append(df['text_split'][i][j])
			# 		label.append(df['label'][i])
			# df_data['content'] = text
			# df_data['label'] =label

			# article = []
			# for i in range(len(df_data)):
			# 	item = {}
			# 	item = {'text':df_data['content'][i],'label':df_data['label'][i]}
			# 	article.append(item)

			df_model = pd.DataFrame(arrayItem)
			train, test = train_test_split(df_model, test_size=0.2,random_state=42)
			tds = Dataset.from_pandas(train)
			vds = Dataset.from_pandas(test)

			dataset = DatasetDict()

			dataset['train'] = tds
			dataset['test'] = vds
			# dataset['train'] = newDatasetTrain
			# dataset['test'] = newDatasetTest
			tokenizer = AutoTokenizer.from_pretrained(current_path.joinpath('sentiment-default'))
			def tokenize_function(examples):
				return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=128)

			tokenized_datasets = dataset.map(tokenize_function, batched=True)
			# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
			# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
			full_train_dataset = tokenized_datasets["train"]
			full_eval_dataset = tokenized_datasets["test"]
			print('length train',len(full_train_dataset))
			print('length test',len(full_eval_dataset))
			print('getmodel')
			config = AutoConfig.from_pretrained(current_path.joinpath(id))
			config.num_labels = 3
			
			config.id2label = {
			0: "NEG",
			1: "POS",
			2: "NEU",
			}
			config.label2id = {
			"POS": 1,
			"NEG": 0,
			"NEU": 2,
			}
			
			model = AutoModelForSequenceClassification.from_config(config)
			# model = AutoModelForSequenceClassification.from_pretrained(current_path.joinpath(id))
			batch_size = 2
			num_epochs = 5
			training_args = TrainingArguments(
				output_dir=current_path.joinpath(id),     # output directory
				per_device_train_batch_size=batch_size,   
				per_device_eval_batch_size=batch_size,                 
				weight_decay=0.01,               
				logging_dir='./logs',  
				logging_strategy = 'steps',        
				logging_steps=100,
				evaluation_strategy="epoch",
				save_strategy='epoch',
				learning_rate=2e-5,
				num_train_epochs=num_epochs,
				logging_first_step = True,
			)
			from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

			def compute_metrics(eval_preds):
				logits, labels = eval_preds
				predictions = np.argmax(logits, axis=-1)
				accuracy = accuracy_score(labels, predictions)
				precision = precision_score(labels, predictions, average='weighted')
				recall = recall_score(labels, predictions, average='weighted')
				f1 = f1_score(labels, predictions, average='weighted')

				metrics_dict = {
					'accuracy': accuracy,
					'precision': precision,
					'recall': recall,
					'f1': f1
				}

				return metrics_dict
			trainer = Trainer(model=model, args=training_args, train_dataset=full_train_dataset,
							eval_dataset=full_eval_dataset, compute_metrics=compute_metrics)
			numberCheckPoint = num_epochs*math.ceil(len(full_train_dataset)/batch_size)
			print('start train')
			trainer.train()
			print('Finish Trainer')
			for obj in trainer.state.log_history:
				print(obj)
			os.chdir(current_path)
			os.rename(id, "rename")
			print('start copy')
			destination = shutil.copytree(current_path.joinpath('rename/checkpoint-'+str(numberCheckPoint)), current_path.joinpath(id))
			# file_paths = [
			# 	'tokenizer_config.json',
			# 	'special_tokens_map.json',
			# 	'added_tokens.json',
			# 	'bpe.codes',
			# 	'vocab.txt'
			# ]
			# for file_path in file_paths:
			# 	shutil.copy2(current_path.joinpath('rename/'+str(file_path)), current_path.joinpath(id))
			remove = shutil.rmtree(current_path.joinpath('rename'))
			print('finish copy')
			time.sleep(15)
			print('evaluate')
			point = trainer.evaluate()
			print(point['eval_accuracy'])
			pointModel = point['eval_accuracy'] * 100
			model = find_model({"_id": ObjectId(id)})
			lastScore = model['score']
			update_status_model({"_id": ObjectId(id)}, {"$set": {"lastScore": lastScore, "score": pointModel, "status": 0}})
		except Exception as e:
				print("ERROR: " + str(e))
				update_status_model({"_id": ObjectId(id)}, {"$set": {"status": 2}})

