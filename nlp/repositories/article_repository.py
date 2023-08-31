from re import T
from shutil import Error
from bson.objectid import ObjectId
from nltk.util import pr
from numpy import number, percentile
from pymongo import MongoClient
import time
import nltk 
from nltk import tokenize
from transformers import BertTokenizer
import re
import os
from dotenv import load_dotenv
load_dotenv()
DB_URL = os.environ.get('DB_URL')
client = MongoClient(DB_URL)
# client = MongoClient("mongodb://crawl02:crawl02123@localhost:27017/?authSource=FinSight")

article_collection = client["FinSight"]["posts"]
tag_collection = client["FinSight"]["tags"]
tagmap_collection = client["FinSight"]["tagmaps"]
model_collection = client["FinSight"]["models"]

def ckeck_database_article_repository():
    global article_collection
    global tag_collection
    global tagmap_collection
    global model_collection

    
    article_collection = client["FinSight"]["posts"]
    tag_collection = client["FinSight"]["tags"]
    tagmap_collection = client["FinSight"]["tagmaps"]
    model_collection = client["FinSight"]["models"]

def get_unprocess_tag_article_ai(language,flagModel):
    ''' Get article which is unproccesed by ai tag '''
    article = article_collection.find_one(
        {
            "$query": {
                "$and": [
                    {"isTagAi": {"$exists": flagModel}},
                    {"languageCrawl": language},
                    {"status": "0"},
                ]
            },
            "$orderby": {"timeCrawlPage": -1},
        }
    )
    return article

def count_unprocess_tag_article_ai(language,flagModel):
    ''' Count article which is unproccesed by ai tag '''
    count = article_collection.count_documents(
        {
            "$and": [
                {"isTagAi": {"$exists": flagModel}},
                {"languageCrawl": language},
                {"status": "0"},
            ]
        }
    )
    return count

def get_unprocess_tag_article_lib(language):
    ''' Find article which is not processed tag '''
    article = article_collection.find_one(
        {
            "$query": {
                "$and": [
                    {"isTag": {"$exists": False}},
                    {"languageCrawl": language},
                    {"status": "0"},
                ]
            },
            "$orderby": {"timeCrawlPage": -1},
        }
    )
    return article

def get_unprocess_classification_article_ai(language):
    ''' Get article which is unproccesed by ai tag '''
    article = article_collection.find_one(
        {
            "$query": {
                "$and": [
                    {"isClassification": {"$exists": False}},
                    {"languageCrawl": language},
                    {"status": "0"},
                ]
            },
            "$orderby": {"timeCrawlPage": -1},
        }
    )
    return article

def count_unprocess_classification_article_ai(language):
    ''' Count article which is unproccesed by ai tag '''
    count = article_collection.count_documents(
        {
            "$and": [
                {"isClassification": {"$exists": False}},
                {"languageCrawl": language},
                {"status": "0"},
            ]
        }
    )
    return count

def update_article(filter, update):
    article_collection.update_one(filter, update)

def update_status_model(filter, update):
    model_collection.update_one(filter, update)

def find_model(filter):
    model = model_collection.find_one(filter)
    return model

def extract_article_verify():
    try:
        article = []
        for x in article_collection.find({ 'status':"0",'classificationStatus':0,"isClassification":True}):
            item = {}
            if x['category'] == 'POS':
                item = {'text_origin':__remove_special_character(x['title']),'label':'POS'}
            if x['category'] == 'NEU':
                item = {'text_origin':__remove_special_character(x['title']),'label':'NEU'}
            if x['category'] == 'NEG':
                item = {'text_origin':__remove_special_character(x['title']),'label':'NEG'}
       
            article.append(item)
        return article
    except Exception as e:
        print("ERROR: " + str(e))
def get_article_verify():
    try:
        article = []
        for x in article_collection.find({ 'status':"0",'classificationStatus':0,"isClassification":True}):
            item = {}
            if x['category'] == 'POS':
                item = {'text':__remove_special_character(x['title']),'label':1}
            if x['category'] == 'NEU':
                item = {'text':__remove_special_character(x['title']),'label':2}
            if x['category'] == 'NEG':
                item = {'text':__remove_special_character(x['title']),'label':0}
       
            article.append(item)
        return article
    except Exception as e:
        print("ERROR: " + str(e))

def get_tag_verify():
    
    lisTag = tagmap_collection.aggregate([
        {"$lookup": {'from': "tags", 'localField': "tag_id", 'foreignField': "_id", 'as': "tags"}},
        {'$unwind': {'path': "$tags",'preserveNullAndEmptyArrays': True}}, {"$match": {'tags.source':"1","tags.tagStatus": 0}},
        {"$project": {"tags.name": 1,"tags.type": 1,}},
        {"$group": {"_id": {"name": "$tags.name","type": "$tags.type",},"count": {'$sum': 1},} }, {"$sort": {"count": -1}},
    ])
    tag = []
    countItem = 0
    countArrayItem = 0
    arrayToken = []
    arrayNerTags = []
    for i in lisTag:
        id = i['_id']
        typeTag = id.get('type', None)
        if typeTag:
            countItem = countItem + 1
            splitWord = i['_id']['name'].split(' ')
            # for j,index in splitWord:
            for index,text in enumerate(splitWord):
                if index == 0 :
                    arrayToken.append(text)
                    if i['_id']['type'] == 'PER' :
                        arrayNerTags.append(2)
                    if i['_id']['type'] == 'ORG' :
                        arrayNerTags.append(4)
                    if i['_id']['type'] == 'LOC' :
                        arrayNerTags.append(6)
                    if i['_id']['type'] == 'MISC' :
                        arrayNerTags.append(8)
                else :
                    arrayToken.append(text)
                    if i['_id']['type'] == 'PER' :
                        arrayNerTags.append(2)
                    if i['_id']['type'] == 'ORG' :
                        arrayNerTags.append(4)
                    if i['_id']['type'] == 'LOC' :
                        arrayNerTags.append(6)
                    if i['_id']['type'] == 'MISC' :
                        arrayNerTags.append(8)
            if countItem == 8 :
                arrayNerTags.append(0)
                arrayToken.append('.')
                countArrayItem = countArrayItem + 1
                
                objItem = {'id': str(countArrayItem) , 'tokens':arrayToken,'pos_tags':[],'chunk_tags':[],'ner_tags':arrayNerTags}
                
                tag.append(objItem)
                objItem = {}
                arrayToken = []
                arrayNerTags = []
                countItem = 0
    return tag
def get_data_article_for_tag(language,id,tokenizer): 
    # get name & type of name (tag)  of trained post 
    querylistArticleTag = tagmap_collection.aggregate([
        {"$lookup": {'from': "tags", 'localField': "tag_id", 'foreignField': "_id", 'as': "tags"}},
        {'$unwind': {'path': "$tags",'preserveNullAndEmptyArrays': True}}, {"$match": {'tags.tagStatus':0}},
        {"$project": {"tags.name": 1,"tags.type": 1,"article_id": 1,}},
        {"$group": {"_id": "$article_id",'name': { "$push": "$tags.name" },'type': { "$push": "$tags.type" }}},
    ])
    listArticle = []
    for i in querylistArticleTag:
        listArticle.append(i)
    arrayNameTag = []
    arrayTypeTag = []
    arrayItemDataset = [] 
    countArrayItem = 0
    countArticleTraining = 0
    arrayIdArticleHasTraining = []
    # lấy 500 bài viết để tạo database

    # get list post has training
    modelTraining = model_collection.find_one({"_id": ObjectId(id)})
    listArticleHasTraining = modelTraining['articleHasTraining']
    arrayIdArticleHasTraining = listArticleHasTraining.copy()

    # get each posts
    for article in article_collection.find({'status':"0",'isTagAi':True,'languageCrawl':language}):
        #check post has training
        flagArticleTraining = False
        for idArticle in listArticleHasTraining:
            if str(idArticle) == str(article['_id']) :
              flagArticleTraining = True
        if flagArticleTraining == True:
            continue
        #post have not training
        else:
            if countArticleTraining > 3200 : #get 500 post
                break
            else:
                countArticleTraining = countArticleTraining + 1
                arrayIdArticleHasTraining.append(str(article['_id'])) 
                arrayNameTag = []
                arrayTypeTag = []
                # check & get name in post with name in trained post
                for itemTag in listArticle:
                    if str(itemTag['_id']) == str(article['_id']) :
                        arrayNameTag = itemTag['name']
                        arrayTypeTag = itemTag['type']
                descriptionArticle = article['description']
                
                splitSentence = tokenize.sent_tokenize(descriptionArticle)
                for sentence in splitSentence:
                    sentence = sentence.replace('/',' / ')
                    splitText = []
                    # splitText = nltk.word_tokenize(sentence)
                    splitText = tokenizer.tokenize(sentence)
                    # mảng text và type của câu 
                    arrayTypeText = []
                    for typeText in splitText : 
                        flagArrayTypeText = False 
                        arrayTypeText.append(0)
                    #get tag for each name in sentence
                    for indexTag,tag in enumerate(arrayNameTag):
                        tag = tag.strip()
                        splitTag=[]
                        # splitTag = nltk.word_tokenize(tag)
                        splitTag = tokenizer.tokenize(tag)
                        lenSplitTag = len(splitTag)
                        for index,text in enumerate(splitText):
                            stringText = ''
                            stringTag = ''
                            for indexText,text in enumerate(splitTag):
                                numberText = index + indexText
                                if numberText <= len(splitText) - 1:
                                    textLower = splitText[numberText]
                                    stringText = stringText +' '+textLower
                                    stringTag = stringTag +' '+splitTag[indexText]
                            stringText = stringText.strip()
                            stringTag = stringTag.strip()
                            if str(stringText) == str(stringTag):
                                for indexText,text in enumerate(splitTag):
                                    numberText = index + indexText
                                    if numberText<=len(splitText) - 1:
                                        typeOfText =  arrayTypeTag[indexTag]
                                        flagArrayTypeText = True
                                        if indexText == 0:
                                            if typeOfText == 'PER' :
                                                arrayTypeText[numberText] = 1
                                            if typeOfText == 'ORG' :
                                                arrayTypeText[numberText] = 3
                                            if typeOfText == 'LOC' :
                                                arrayTypeText[numberText] = 5
                                            if typeOfText == 'MISC' :
                                                arrayTypeText[numberText] = 7
                                        else:
                                            if typeOfText == 'PER' :
                                                arrayTypeText[numberText] = 2
                                            if typeOfText == 'ORG' :
                                                arrayTypeText[numberText] = 4
                                            if typeOfText == 'LOC' :
                                                arrayTypeText[numberText] = 6
                                            if typeOfText == 'MISC' :
                                               arrayTypeText[numberText] = 8
                    if flagArrayTypeText == True:
                        countArrayItem = countArrayItem + 1

                        ObjItemDataset = {'id': str(countArrayItem) , 'tokens':splitText,'pos_tags':[],'chunk_tags':[],'ner_tags':arrayTypeText}
                        arrayItemDataset.append(ObjItemDataset)
    print('Get data ok')
    return [arrayItemDataset,arrayIdArticleHasTraining]
def get_data_article_for_evaluate_tag(): 
    # get name & type of name (tag)  of trained post 
    querylistArticleTag = tagmap_collection.aggregate([
        {"$lookup": {'from': "tags", 'localField': "tag_id", 'foreignField': "_id", 'as': "tags"}},
        {'$unwind': {'path': "$tags",'preserveNullAndEmptyArrays': True}}, {"$match": {'tags.tagStatus':0}},
        {"$project": {"tags.name": 1,"tags.type": 1,"article_id": 1,}},
        {"$group": {"_id": "$article_id",'name': { "$push": "$tags.name" },'type': { "$push": "$tags.type" }}},
    ])
    listArticle = []
    for i in querylistArticleTag:
        listArticle.append(i)
    arrayNameTag = []
    arrayTypeTag = []
    arrayItemDataset = [] 
    countArrayItem = 0
    countArticleTraining = 0

    # get each posts
    for article in article_collection.find({'status':"0",'isTagAi':True,}):       
        countArticleTraining = countArticleTraining + 1
        
        arrayNameTag = []
        arrayTypeTag = []
        # check & get name in post with name in trained post
        for itemTag in listArticle:
            if str(itemTag['_id']) == str(article['_id']) :
                arrayNameTag = itemTag['name']
                arrayTypeTag = itemTag['type']
        descriptionArticle = article['content']
        
        splitSentence = descriptionArticle.split(".")
        for sentence in splitSentence:
            sentence = __remove_special_character(sentence)
            splitText = []
            # splitText = nltk.word_tokenize(sentence)
            # splitText = tokenizer.tokenize(sentence)
            # splitText = sentence.split(" ")
            splitText = [word for word in sentence.split(" ") if len(word)>1]
            # mảng text và type của câu 
            arrayTypeText = []
            for typeText in splitText : 
                flagArrayTypeText = False 
                arrayTypeText.append(8)
            #get tag for each name in sentence
            for indexTag,tag in enumerate(arrayNameTag):
                tag = tag.strip()
                splitTag=[]
                tag = __remove_special_character(tag)
                # splitTag = tag.split(" ")
                splitTag= [word for word in tag.split(" ") if word]
                lenSplitTag = len(splitTag)
                for index,text in enumerate(splitText):
                    stringText = ''
                    stringTag = ''
                    for indexText,text in enumerate(splitTag):
                        numberText = index + indexText
                        if numberText <= len(splitText) - 1:
                            textLower = splitText[numberText]
                            stringText = stringText +' '+textLower
                            stringTag = stringTag +' '+splitTag[indexText]
                    stringText = stringText.strip()
                    stringTag = stringTag.strip()
                    if str(stringText) == str(stringTag):
                        for indexText,text in enumerate(splitTag):
                            numberText = index + indexText
                            if numberText<=len(splitText) - 1:
                                typeOfText =  arrayTypeTag[indexTag]
                                flagArrayTypeText = True
                                if indexText == 0:
                                    if typeOfText == 'ORG' :
                                        arrayTypeText[numberText] = 2
                                    if typeOfText == 'LOC' :
                                        arrayTypeText[numberText] = 0
                                    if typeOfText == 'PER' :
                                        arrayTypeText[numberText] = 3
                                    if typeOfText == 'MISC' :
                                        arrayTypeText[numberText] = 1
                                else:
                                    if typeOfText == 'ORG' :
                                        arrayTypeText[numberText] = 6
                                    if typeOfText == 'LOC' :
                                        arrayTypeText[numberText] = 4
                                    if typeOfText == 'PER' :
                                        arrayTypeText[numberText] = 7
                                    if typeOfText == 'MISC' :
                                        arrayTypeText[numberText] = 5
            if flagArrayTypeText == True:
                
                if len(arrayTypeText) >0 :
                    countArrayItem = countArrayItem + 1
                    ObjItemDataset = {'id': str(countArrayItem) , 'tokens':splitText,'pos_tags':[],'chunk_tags':[],'ner_tags':arrayTypeText}
                    arrayItemDataset.append(ObjItemDataset)
    print('Get data ok')
    print(len(arrayItemDataset))
    return arrayItemDataset


def __remove_special_character(text):
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(",", " ")
    text = text.replace(".", " ")
    text = text.replace("/", " ")
    text = text.replace("”", " ")
    text = text.replace("“", " ")
    text = text.replace(":", " ")
    text = text.replace("'"," ")
    text = text.replace('"'," ")
    text = text.replace('’'," ")
    text = text.replace("'"," ")
    text = text.replace("+", " ")
    text = text.replace("–", " ")
    text = text.replace("%", " ")
    text = text.replace('\r\n'," ")
    text = text.replace('\n'," ")
    text = text.replace('\r'," ")
    text = text.replace(']'," ")
    text = text.replace('['," ")
    text = text.replace("-","")
    text = text.replace("\\ufeff", "")
    text = text.replace('\\'," ")
    text = re.findall(r'\b\w+\b', text, re.UNICODE)
    text = ' '.join(text)
    
    text = re.sub(r'\s{3,}', ' ', text)
    text = text.strip()
    return text

