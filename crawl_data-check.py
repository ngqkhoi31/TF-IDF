import numpy as np
import sklearn
import underthesea
import json
import os
from tqdm.autonotebook import tqdm
import glob
from pprint import pprint
import nltk
from underthesea import word_tokenize
from sklearn.decomposition import TruncatedSVD
import math
import sklearn
import sklearn.metrics

#duong dan toi thu muc chua tep json
directory = r"D:\University\Fourth-year Collage\Clould Computing\AI_Search_TF IDF\Test TF_IDF\dataset"

#danh sanh de luu tru cac muc tu cac tap json
items = []

#duyet qua tat ca cac tep co duoi json trong thu muc
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            items += json.load(f)
            
    #if filename.endswith(".txt"):
    #    filepath = os.path.join(directory, filename)
    #    with open(filepath, "r", encoding="utf-8") as f:
    #        text_content = f.read()
    #        items.append({"text": text_content})

#kiem tra va in
if items:
    print(items[0])
else:
    print("No items found.")
    
    
def word_tokenizer_item(item):
    '''
    Word tokenize for all field in a item
    '''
    for key in item.keys():
        item[key] = word_tokenize(item[key], format='text')
    return item
def count_unigram(text):
    '''
    Count appearance number for each vocabulary
    '''
    counter = {}
    words = text.split()
    vocabs = set(words)
    for vocab in vocabs:
        if not vocab.isdigit():
            counter[vocab] = words.count(vocab)
    return counter

def combine_metadata(item):
    newItem = {}
    newItem['name'] = item['title']
    newItem['desc'] = ""
    for key in item.keys():
        newItem["desc"] += f" . {item[key]}"
    return newItem
    
def count_word_in_dataset(items):
    '''
     Thống kê số lần xuất hiện từng từ trên toàn bộ dataset
    '''   
    nameCounter = {}
    descCounter = {}
    for item in items:
        for word in item['unigram_name'].keys():
            if word in nameCounter.keys():
                nameCounter[word] += 1
            else:
                nameCounter[word] = 1
        for word in item['unigram_desc'].keys():
            if word in descCounter.keys():
                descCounter[word] += 1
            else:
                descCounter[word] = 1
    return nameCounter, descCounter

newItems = []
for item in tqdm(items):
    newItem = word_tokenizer_item(item)
    newItem = combine_metadata(newItem)
    newItem['unigram_name'] = count_unigram(newItem['name'])
    newItem['unigram_desc'] = count_unigram(newItem['desc'])
    newItems.append(newItem)

def tfidf(doc_len, corpus_len, doc_counter, corpus_counter, k=2):
    vector_len = len(corpus_counter)
    tfidf_vector = np.zeros((vector_len,))
    for i, key in enumerate(corpus_counter.keys()):
        if key in doc_counter.keys():
            tf = (k+1)*doc_counter[key]/(k+doc_counter[key])
            idf = math.log((corpus_len+1)/(corpus_counter[key]))
            tfidf_vector[i] = tf*idf
    return tfidf_vector

nameCounter, descCounter = count_word_in_dataset(newItems)
tfidf_vectors = []
corpus_len = len(newItems)
for item in tqdm(newItems):
    doc_len = len(item['desc'])
    tfidf_vectors.append(
        tfidf(doc_len, corpus_len, item['unigram_desc'], descCounter)
    )
tfidf_vectors = np.array(tfidf_vectors) # dam bao tfidf_vectors laf mang 2d

svd = TruncatedSVD(n_components=256)
svd.fit(tfidf_vectors)
svd_tfidf_vexter = svd.transform(tfidf_vectors)

def process_query(query, descCounter, corpus_len):
    #truy van
    tokenized_query = word_tokenize(query, format='text')
    
    #count unigram in query
    query_unigram = count_unigram(tokenized_query)
    
    #calculate tfidf vector for query
    query_vector = tfidf(len(tokenized_query.split()), corpus_len, query_unigram, descCounter)
    return query_vector


new_query = "trưởng phòng quản lý trải nghiệm khách hàng"
#xu ly query moi
query_vector = process_query(new_query, descCounter, corpus_len)
query_vector = np.reshape(query_vector, (1, -1))

#seach
sim_maxtrix = sklearn.metrics.pairwise.cosine_similarity(query_vector, tfidf_vectors)
sim_maxtrix = np.reshape(sim_maxtrix, (-1))

idx = (-sim_maxtrix).argsort()[:20]
for _id in idx:
    print(_id, sim_maxtrix[_id])
    print(newItems[_id]['name'].upper())
    print(newItems[_id]['desc'], "\n\n")