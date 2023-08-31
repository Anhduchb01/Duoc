import time
import datetime
from urllib import request
import pytz
from pymongo import MongoClient
from bson.objectid import ObjectId
from slugify import slugify
from ..utilities.datetime_ultility import get_time_article
from ..repositories.article_repository import ckeck_database_article_repository,get_unprocess_tag_article_lib, get_unprocess_tag_article_ai, count_unprocess_tag_article_ai, update_article , find_model,update_status_model , get_tag_verify ,get_data_article_for_tag,get_data_article_for_evaluate_tag
from ..services.ai.nlp.bert_model import generate_keyword, split_sentence, split_sentence_create ,initialization_model,initialization_model_evaluate ,generate_keyword_default
from .ai.nlp.tags_lib_model import generate_lib_tag
from datasets import load_dataset ,load_metric ,Dataset,  DatasetDict
import pandas as pd
from transformers import AutoTokenizer , DataCollatorForTokenClassification , TrainingArguments, AutoModelForTokenClassification , AutoConfig , AutoModelForSequenceClassification
import numpy as np
import shutil
import os
import time
from pathlib import Path
import math
import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from evaluate import evaluator
from datasets import Sequence, Value ,ClassLabel
from underthesea import ner
current_path = Path(__file__).parent.parent.joinpath('ai_model')
import re
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.environ.get('DB_URL')
client = MongoClient(DB_URL)

# client = MongoClient("mongodb://crawl02:crawl02123@localhost:27017/?authSource=FinSight")

tags_collection = client["FinSight"]["tags"]
tagmap_collection = client["FinSight"]["tagmaps"]
model_collection = client["FinSight"]["models"]
historyGenerateTag_collection = client["FinSight"]["historygeneratetags"]
article_collection = client["FinSight"]["posts"]
historyTag_collection = client["FinSight"]["taghistorys"]

def ckeck_database_tag_service(page):
	ckeck_database_article_repository()
	global tags_collection
	global tagmap_collection
	global model_collection
	global historyGenerateTag_collection
	global article_collection
	global historyTag_collection
	
	tags_collection = client["FinSight"]["tags"]
	tagmap_collection = client["FinSight"]["tagmaps"]
	model_collection = client["FinSight"]["models"]
	historyGenerateTag_collection = client["FinSight"]["historygeneratetags"]
	article_collection = client["FinSight"]["posts"]
	historyTag_collection = client["FinSight"]["taghistorys"]


def predict_tag_lib(text):
	
	tags = generate_lib_tag(text)

	return tags
   
def process_tag_lib():
	''' Use libraries to generate tag '''
	timeExcute = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	timeExcute = str(timeExcute)
	listTag = []
	listTagMap = []
	listArticleHasProcessed = []
	# Update Model status to processing
	 
	
	model_collection.update_one({"name": "Library (Underthesea)"}, {"$set": {"status": 1}})
	try :
		findModelBase = historyGenerateTag_collection.find_one({"model_id": 'Library (Underthesea)'})
		if findModelBase:
			arrayArticleHasProcessed = findModelBase['listArticleHasProcessed']
			print('model da ton tai')
		else:
			arrayArticleHasProcessed = []
			historyGenerateTag_collection.insert_one({ "time": timeExcute, "model_id": 'Library (Underthesea)','listArticleHasProcessed':listArticleHasProcessed})
	
		cursor = article_collection.find({'status':"0"},no_cursor_timeout=True)
		for article in tqdm(cursor):
			flagArticle = True
			for articleHasProcessed in arrayArticleHasProcessed:
				if str(articleHasProcessed['article_id']) == str(article['_id']):
					flagArticle = False
			
			if flagArticle == False:
				continue
			# Get unprocess article
			# article = get_unprocess_tag_article_lib(language)
			# if exist article
			if article:
				text = article["content"] # Get content of Article
				if text == None or text == '':
						continue
				# If exist text
				if text != None or text != '':
					# print("PROCESS: https://finsight.sinka.vn/detail-new/" + str(article["_id"]) + " " + "-" + " " + str(datetime.datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))), flush=True)
					# Try get tag from libaries, then save result to database
					try:
						year = get_time_article(article) # Get year of article
						# Generate tag by language
						tags = []	
						text_process = split_sentence(text)
						tags_array = []
						tags_name_array = []
						for sentence in text_process:
							tags = generate_lib_tag(sentence)
							for tag in tags:
								if tag["name"] not in tags_name_array:
									tags_array.append(tag)
									tags_name_array.append(tag["name"])
							for tag in tags_array:
								for tag1 in tags_array:
									if tag["name"] in tag1["name"] and len(tag["name"]) < len(tag1["name"]):
										tags_array.remove(tag1)
						for tag in tags_array:
							listTagMapHistory = []
							tag_exit = historyTag_collection.find_one({"model_id": 'Library (Underthesea)','name':tag['name']})
							if tag_exit:
								newScore = None
								listTagMapHistory = tag_exit['listTagMap']
								listTagMapHistory.append({ "article_id": article["_id"], "year": year})
								historyTag_collection.update_one({"model_id": 'Library (Underthesea)','name':tag['name']}, {"$set": {"listTagMap": listTagMapHistory,"score": newScore}})
							else:
								slug_tag = slugify(tag["name"])
								score = None
								listTagMapHistory.append({ "article_id": article["_id"], "year": year})
								historyTag_collection.insert_one({'model_id':'Library (Underthesea)', "name": tag["name"], "slug": slug_tag, "type": tag["type"], "score": score,"listTagMap": listTagMapHistory})
						
					
						listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": 'Library (Underthesea)'})
						arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
						arrayArticleHasProcessed.append({'article_id':article["_id"],'isTag':True})
						historyGenerateTag_collection.update_one({"model_id": 'Library (Underthesea)'}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
						
					except Exception as e:
						print("ERROR: " + str(e) + " https://finsight.vn/detail-new/" + str(article["_id"]) + " " + "-" + " " + str(datetime.datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))), flush=True)
						listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": 'Library (Underthesea)'})
						arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
						arrayArticleHasProcessed.append({'article_id':article["_id"],'isTag':False})
						historyGenerateTag_collection.update_one({"model_id": 'Library (Underthesea)'}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
						
				else:
					
					listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": 'Library (Underthesea)'})
					arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
					arrayArticleHasProcessed.append({'article_id':article["_id"],'isTag':False})
					historyGenerateTag_collection.update_one({"model_id": 'Library (Underthesea)'}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
					
				time.sleep(1)
			else:
				break
		cursor.close()
		# Update status of executing generate tag
	
		listTag = historyTag_collection.find({'model_id':'Library (Underthesea)'})
		model_old = model_collection.find_one({"name": "Library (Underthesea)"})
		total = len(list(listTag))
		lastTotalTag = model_old['totalTag']

		model_collection.update_one({"name": "Library (Underthesea)"}, {"$set": {"status": 0,'totalTag':total,'lastTotalTag':lastTotalTag}})
		return {"data": "finish"}
	except Exception as e:
            print("ERROR: " + str(e)) 
            model_collection.update_one({"name": "Library (Underthesea)"}, {"$set": {"status": 2}})
            
           

   

	
def extract_data_for_tag():
	print('start extract data')
	data = get_data_article_for_evaluate_tag()
	dfdata = pd.DataFrame(data)
	dsdata = Dataset.from_pandas(dfdata)
	return data
def evaluate_tag_ai(language, id):
	
	print('start evaluate')
	tokenizer = AutoTokenizer.from_pretrained(current_path.joinpath(id))
	data = get_data_article_for_evaluate_tag()
	dfdata = pd.DataFrame(data)
	dsdata = Dataset.from_pandas(dfdata)
	print('load data ok ')
	new_features = dsdata.features.copy()
	new_features["tokens"] = Sequence(feature=Value(dtype="string"))

	new_features["ner_tags"] = Sequence(feature=ClassLabel(names=[   
					'O',
					'B-PER',  
					'I-PER',
					'B-ORG',  
					'I-ORG',
					'B-LOC',  
					'I-LOC' ,
					'B-MISC', 
					'I-MISC',  

					]))
	dsdata = dsdata.cast(new_features)
	model_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": 1}})

	ner = initialization_model_evaluate(id)
	task_evaluator = evaluator("token-classification")

	results = task_evaluator.compute(
			model_or_pipeline=ner,
			data=dsdata,
			metric="seqeval",
			input_column='tokens',
			label_column='ner_tags',

			)
	accuracy = results['overall_accuracy']*100
	model_old = model_collection.find_one({"_id": ObjectId(id)})
	score_old = model_old['score']
	model_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": 0,"lastScore":score_old,"score":accuracy}})
	
	
	return [score_old,accuracy]
	
	
def predict_tag_ai(language, id,timeModel,text):
	
	
	
	ner = initialization_model(id)
	print('load model ok')
	if len(text.split())>0:                    
		text_process = split_sentence(text,language)
		
		print("split text ok")
		for sentence in text_process:
				print('start predict')
				tags = generate_keyword(sentence, ner,language)
				print('predict ok')  
				tags_array = []
				tags_name_array = []
				for sentence in text_process:
					tags = generate_keyword(sentence, ner,language)
					for tag in tags:
						if tag["name"] not in tags_name_array:
							tags_array.append(tag)
							tags_name_array.append(tag["name"])
					for tag in tags_array:
						for tag1 in tags_array:
							if tag["name"] in tag1["name"] and len(tag["name"]) < len(tag1["name"]):
								tags_array.remove(tag1)       
	
			
	
	return tags_array
def process_tag_ai(id,timeModel):
	''' Use Bert model to generate tag '''
	# historyGenerate = historyGenerateTag_collection.find_one({'model_id':str(id)})
	# timeCrawlPage = dayjs().format("YYYY/MM/DD h:mm:ss")
	timeExcute = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
	timeExcute = str(timeExcute)
	listTag = []
	listTagMap = []
	listArticleHasProcessed = []
	arrayArticleHasProcessed = []
	try : 

		if id == 'default':
		
			model_collection.update_one({"name": "AI NER Base"}, {"$set": {"status": 1}})
			findModelBase = historyGenerateTag_collection.find_one({"model_id": str(id)})
			if findModelBase:
				print('model da ton tai')
			else:
				historyGenerateTag_collection.insert_one({ "time": timeExcute, "model_id": str(id),'listArticleHasProcessed':listArticleHasProcessed}) 
			
			ner = initialization_model(id)
			listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id)})
			arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
			print('load model default -ok')
			cursor  =article_collection.find({'status':"0"},no_cursor_timeout=True).batch_size(20)
			for article in tqdm(cursor):
				flagArticle = True
				for articleHasProcessed in arrayArticleHasProcessed:
					if str(articleHasProcessed['article_id']) == str(article['_id']):
						flagArticle = False
						break
				if flagArticle == False:
					continue
				if article:
					
					text = article["content"]
					if text == None or text == '':
						break
					year = get_time_article(article)
					if text != None or text != '':
						# split big sentences to small sentences. 
						text_process = split_sentence(text)
						try:
							tags_array = []
							tags_name_array = []
							for sentence in text_process:
								tags = generate_keyword_default(sentence, ner)
								for tag in tags:
									if tag["name"] not in tags_name_array:
										tags_array.append(tag)
										tags_name_array.append(tag["name"])
								for tag in tags_array:
									for tag1 in tags_array:
										if tag["name"] in tag1["name"] and len(tag["name"]) < len(tag1["name"]):
											tags_array.remove(tag1)

								
							# Insert tags
							for tag in tags_array:
		
								listTagMapHistory = []
								tag_exit = historyTag_collection.find_one({"model_id": str(id),'name':tag['name']})
								# listTagHistory = findModelHistory['listTag']
								# listTagMapHistory = []
								# tag_exit = ''
								# slug = ''
								# lastScore = 0
								# for tagInListTag in listTagHistory:
								#     checkTag = tagInListTag.get('name', None)
								#     if checkTag:
								#         if tagInListTag['name'] == tag["name"]:
								#             tag_exit = tag
								#             lastScore = tagInListTag['score']
								#             listTagMapHistory = tagInListTag['listTagMap']
								#             slug = tagInListTag['slug']
								#             break  

								if tag_exit:
									
									newScore = (float(tag["score"]) + float(tag_exit['score']) )/2
									listTagMapHistory = tag_exit['listTagMap']
									listTagMapHistory.append({ "article_id": article["_id"],  "year": year})
									historyTag_collection.update_one({"model_id": str(id),'name':tag['name']}, {"$set": {"listTagMap": listTagMapHistory,"score": newScore}})
								else:
									slug_tag = slugify(tag["name"])
									score = float(tag["score"])
									listTagMapHistory.append({ "article_id": article["_id"],  "year": year})
				
									historyTag_collection.insert_one({'model_id':str(id), "name": tag["name"], "slug": slug_tag, "type": tag["type"], "score": score,"listTagMap": listTagMapHistory})
								# tag_exit_goc = tags_collection.find_one({"name": tag["name"], 'source': '1'})
								# # if tag_exit_goc:
								#     tagmap_collection.insert_one({"article_id": article["_id"], "tag_id": tag_exit_goc["_id"],  "year": year})
								#     continue
								# slug_tag = slugify(tag["name"])
								# tag_id = tags_collection.insert_one({"name": tag["name"], "slug": slug_tag, "type": tag["type"], "score": float(tag["score"]), "source": '1'})
								# tagmap_collection.insert_one({"article_id": article["_id"], "tag_id": tag_id.inserted_id, "year": year})
							listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id)})
							arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
							arrayArticleHasProcessed.append({'article_id':article["_id"],'isTagAi':True})
							historyGenerateTag_collection.update_one({"model_id": str(id)}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
						except Exception as e:
							print("ERROR: " + str(e))
							# if error when generate tagai update field isTagAi is False
							listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id)})
							arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
							arrayArticleHasProcessed.append({'article_id':article["_id"],'isTagAi':False})
							historyGenerateTag_collection.update_one({"model_id": str(id)}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
					else:
						listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id)})
						arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
						arrayArticleHasProcessed.append({'article_id':article["_id"],'isTagAi':False})
						historyGenerateTag_collection.update_one({"model_id": str(id)}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
					time.sleep(1)
				else:
					break
			cursor.close()
			print('predict - ok')
			listTag = historyTag_collection.find({"model_id": id})
			total = len(list(listTag))
			model_collection.update_one({"name":"AI NER Base"}, {"$set": {"status": 0,'totalTag':total}})		
		else:
			model_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": 1}})
			findModelBase = historyGenerateTag_collection.find_one({"model_id": str(id),'time':timeModel})
			if findModelBase:
				arrayArticleHasProcessed = findModelBase['listArticleHasProcessed']
				timeExcute=findModelBase['time']
				print('model da ton tai')
			else:
				timeExcute = timeModel
				historyGenerateTag_collection.insert_one({ "time": timeModel, "model_id": str(id),'listArticleHasProcessed':listArticleHasProcessed})
			model = initialization_model(id)
			tokenizer = AutoTokenizer.from_pretrained(current_path.joinpath('ner-default'))
			print('load model - ok')
			cursor  =article_collection.find({'status':"0"},no_cursor_timeout=True).batch_size(20)
			for article in tqdm(cursor):
				if len(arrayArticleHasProcessed) != 0:
					flagArticle = True
					for articleHasProcessed in arrayArticleHasProcessed:
						if str(articleHasProcessed['article_id']) == str(article['_id']):
							flagArticle = False
							
					if flagArticle == False:
						continue
				if article:
					text = article["content"]
					if len(text.split())==0:
						print('id None Articale',str(article['_id']))
						continue
					year = get_time_article(article)
					if text:
						# split big sentences to small sentences. 
						text_process = split_sentence_create(text)
						try:
							tags_array = []
							tags_name_array = []
							for sentence in text_process:
								tags = generate_keyword(sentence, tokenizer, model)
								for tag in tags:
									if tag["name"] not in tags_name_array:
										tags_array.append(tag)
										tags_name_array.append(tag["name"])
								for tag in tags_array:
									for tag1 in tags_array:
										if tag["name"] in tag1["name"] and len(tag["name"]) < len(tag1["name"]):
											tags_array.remove(tag1)
							# Insert tags
							for tag in tags_array:

								listTagMapHistory = []
								tag_exit = historyTag_collection.find_one({"model_id": str(id),'name':tag['name'],"time": timeExcute})
								if tag_exit:
									newScore = (float(tag["score"]) + float(tag_exit['score']) )/2
									listTagMapHistory = tag_exit['listTagMap']
									listTagMapHistory.append({ "article_id": article["_id"], "year": year})
									historyTag_collection.update_one({"model_id": str(id),'name':tag['name']}, {"$set": {"listTagMap": listTagMapHistory,"score": newScore}})
								else:
									slug_tag = slugify(tag["name"])
									score = float(tag["score"])
									listTagMapHistory.append({ "article_id": article["_id"], "year": year})
									historyTag_collection.insert_one({'model_id':str(id),"time": timeExcute, "name": tag["name"], "slug": slug_tag, "type": tag["type"], "score": score,"listTagMap": listTagMapHistory})
								
							# Update article TagAI status
							listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id),"time": timeExcute})
							arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
							arrayArticleHasProcessed.append({'article_id':article["_id"],'isTagAi':True})
							historyGenerateTag_collection.update_one({"model_id": str(id),"time": timeExcute}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
							# update_article({"_id": article["_id"]}, {"$set": {"isTagAi": True}})
						except Exception as e:
							print("ERROR: " + str(e))
							# if error when generate tagai update field isTagAi is False
							listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id),"time": timeExcute})
							arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
							arrayArticleHasProcessed.append({'article_id':article["_id"],'isTagAi':False})
							historyGenerateTag_collection.update_one({"model_id": str(id),"time": timeExcute}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
					else:
						listArticleHasProcessed = historyGenerateTag_collection.find_one({"model_id": str(id),"time": timeExcute})
						arrayArticleHasProcessed = listArticleHasProcessed['listArticleHasProcessed']
						arrayArticleHasProcessed.append({'article_id':article["_id"],'isTagAi':False})
						historyGenerateTag_collection.update_one({"model_id": str(id),"time": timeExcute}, {"$set": {"listArticleHasProcessed": arrayArticleHasProcessed}})
					time.sleep(1)
				else:
					break
			cursor.close()
			print('predict - ok')
		
			listTag = historyTag_collection.find({"model_id": id})
			model_old = model_collection.find_one({"_id": ObjectId(id)})
			total = len(list(listTag))
			lastTotalTag = model_old['totalTag']

			model_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": 0,'totalTag':total,'lastTotalTag':lastTotalTag}})
	except Exception as e:
            print("ERROR: " + str(e))
            if id == 'default':
                model_collection.update_one({"name": "AI NER Base"}, {"$set": {"status": 2}})
            
            else:
                model_collection.update_one({"_id": ObjectId(id)}, {"$set": {"status": 2}})


def run_process_training_tag( id):
	timeExcute = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
	timeExcute = str(timeExcute)
	arrayIdArticleHasTraining = []

	timeExcute = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
	timeExcute = str(timeExcute)
	arrayIdArticleHasTraining = []

	
	update_status_model({"_id": ObjectId(id)}, {"$set": {"status": 1}})
	try:
	# try:
		raw_datasets = DatasetDict()
		arrayItem = get_data_article_for_evaluate_tag()
		df = pd.DataFrame(arrayItem)
		print(df.head())
		train, test = train_test_split(df, test_size=0.2,random_state=42)
		tds = Dataset.from_pandas(train)
		vds = Dataset.from_pandas(test)

		raw_datasets = DatasetDict()

		raw_datasets['train'] = tds
		raw_datasets['validation'] = vds
		label_names = [
				"B-LOCATION",
				"B-MISCELLANEOUS",
				"B-ORGANIZATION",
				"B-PERSON",
				"I-LOCATION",
				"I-MISCELLANEOUS",
				"I-ORGANIZATION",
				"I-PERSON",
				"O"
			]
		id2label = {str(i): label for i, label in enumerate(label_names)}
		label2id = {v: k for k, v in id2label.items()}
		print('start training')
		
		tokenizer = AutoTokenizer.from_pretrained(current_path.joinpath('ner-default'))
		def tokenize_and_align_labels(examples):
			max_seq_len =128
			seq_len = len(examples["tokens"])
			
			sentence = ' '.join(examples["tokens"])
			tag_ids = examples["ner_tags"]
			encoding = tokenizer(sentence,
								padding='max_length',
								truncation=True,
								max_length=max_seq_len)
			subwords = examples["tokens"]
			
			valid_ids = np.zeros(len(encoding.input_ids), dtype=int)
			label_marks = np.zeros(len(encoding.input_ids), dtype=int)
			valid_labels = np.ones(len(encoding.input_ids), dtype=int) * -100
			i = 1
			for idx, subword in enumerate(subwords[:max_seq_len-2]):
				if idx != 0 and subwords[idx-1].endswith("@@"):
					continue
				valid_ids[idx+1] = 1
				valid_labels[idx+1] = tag_ids[i-1]
				i += 1
			if max_seq_len >= seq_len:
				label_padding_size = (max_seq_len - seq_len)
				label_marks[:seq_len] = [1] * seq_len
				tag_ids.extend([0] * label_padding_size)
			else:
				tag_ids = tag_ids[:max_seq_len]
				label_marks[:-2] = [1] * (max_seq_len - 2)
				tag_ids[-2:] = [0] * 2

			items = {key: val for key, val in encoding.items()}
			items['labels'] = valid_labels
			items['valid_ids'] = valid_ids
			items['label_masks'] =  valid_ids
			return items

			
		tokenized_datasets = raw_datasets.map(tokenize_and_align_labels,remove_columns=raw_datasets["train"].column_names)
		metric = load_metric("seqeval")
		def compute_metrics(p):
			predictions, labels = p
			predictions = np.argmax(predictions, axis=2)

			# Remove ignored index (special tokens)
			true_predictions = [
				[label_names[p] for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]
			true_labels = [
				[label_names[l] for (p, l) in zip(prediction, label) if l != -100]
				for prediction, label in zip(predictions, labels)
			]

			results = metric.compute(predictions=true_predictions, references=true_labels)
			flattened_results = {
				"overall_precision": results["overall_precision"],
				"overall_recall": results["overall_recall"],
				"overall_f1": results["overall_f1"],
				"overall_accuracy": results["overall_accuracy"],
			}
			for k in results.keys():
				if(k not in flattened_results.keys()):
					flattened_results[k+"_f1"]=results[k]["f1"]
			return flattened_results
		model = AutoModelForTokenClassification.from_pretrained(current_path.joinpath('ner-default'),num_labels=len(id2label))
		batch_size =2
		number_epochs = 5
		args = TrainingArguments(
			output_dir=current_path.joinpath(id),
			evaluation_strategy="epoch",
			logging_strategy = 'epoch',
			save_strategy="epoch",
			learning_rate=2e-5,
			num_train_epochs=number_epochs,
			weight_decay=0.01,
			logging_dir='./logs',           
			logging_steps=100,
			logging_first_step = True,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,  
		)


		data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
		from transformers import Trainer
		print('start trainer')
		print('length of tokenized_datasets["train"]',len(tokenized_datasets["train"]))
		print('length of tokenized_datasets["validation"]',len(tokenized_datasets["validation"]))
		trainer = Trainer(
			model=model,
			args=args,
			train_dataset=tokenized_datasets["train"],
			eval_dataset=tokenized_datasets["validation"],
			data_collator=data_collator,
			compute_metrics=compute_metrics,
			tokenizer=tokenizer,

		)

		numberCheckPoint =number_epochs* math.ceil(len(tokenized_datasets["train"])/batch_size)
		trainer.train()
		for obj in trainer.state.log_history:
			print(obj)
		print('Train Finish')
		os.chdir(current_path)
		os.rename(id, "rename")
		destination = shutil.copytree(current_path.joinpath('rename/checkpoint-'+str(numberCheckPoint)), current_path.joinpath(id))
		# remove = shutil.rmtree(current_path.joinpath('rename'))

		print('evaluate')
		point = trainer.evaluate()
		print(point['eval_accuracy'])
		pointModel = point['eval_accuracy'] * 100
		model = find_model({"_id": ObjectId(id)})
		lastScore = model['score']
		update_status_model({"_id": ObjectId(id)}, {"$set": {"lastScore": lastScore,'time':timeExcute, "score": pointModel, "status": 0,"articleHasTraining":arrayIdArticleHasTraining}})
	
					



	except Exception as e:
		print("ERROR: " + str(e))
		update_status_model({"_id": ObjectId(id)}, {"$set": {"status": 2}})
	