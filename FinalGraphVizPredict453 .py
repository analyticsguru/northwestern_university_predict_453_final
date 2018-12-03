
# coding: utf-8

# Ishmael Amin
# Predict Analytics 453
# Fall 2017
# 
# # Goal 
# My goal is to read Amazon reviews and metadata and build a dataset that can be parsed and loaded into a graph database.

# In[1]:


import json
import os
import codecs
import pandas as pd


# # Define file paths

# In[62]:


data_directory = os.path.join('.', 'GitHub',
                              'Data')

grocery_gz_input = os.path.join(data_directory,
                                   'reviews_Grocery_and_Gourmet_Food_5.json.gz')

metadata_gz_input = os.path.join(data_directory,
                                   'meta_Grocery_and_Gourmet_Food.json.gz')

grocery_json_raw = os.path.join(data_directory,
                                   'reviews_Grocery_and_Gourmet_Food_5_raw.json')

grocery_json_flattened = os.path.join(data_directory,
                                      'reviews_Grocery_and_Gourmet_Food_flattened.json')

grocery_json_sample = os.path.join(data_directory,
                                      'reviews_Grocery_and_Gourmet_Food_sample.json')

metadata_json_sample = os.path.join(data_directory,
                                      'meta_Grocery_and_Gourmet_Food_sample.json')
                
metadata_json_raw = os.path.join(data_directory,
                                   'metadata_Grocery_and_Gourmet_Food_5_raw.json')

metadata_json_flattened = os.path.join(data_directory,
                                      'meta_Grocery_and_Gourmet_Food_flattened.json')

pickledResults = os.path.join(data_directory,
                                      'PickledResults.pkl')


# In[3]:


import json
import gzip

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

def createjson(in_path,out_path):
    f = open(out_path, 'w')
    for l in parse(in_path):
        f.write(l + '\n')
        #f.write(l + '\n')
    #f.close()


# In[18]:


get_ipython().run_cell_magic('time', '', '\ncreatejson(grocery_gz_input,grocery_json_raw)\n\ncreatejson(metadata_gz_input,metadata_json_raw)')


# # Review Data Structure
# 
# asin - ID of the product, e.g. 0000031852 <br/>
# title - name of the product<br/>
# price - price in US dollars (at time of crawl)<br/>
# imUrl - url of the product image<br/>
# related - related products (also bought, also viewed, bought together, buy after viewing)<br/>
# salesRank - sales rank information<br/>
# brand - brand name<br/>
# categories - list of categories the product belongs to<br/>
# 

# In[5]:


def getFirstLineJSON(path):
    with codecs.open(path, encoding='utf_8') as file:
        first_customer_record = file.readline() 
        return json.loads(first_customer_record)
#j = json.loads(json.dumps(jsonStr)) with j = json.loads(jsonStr)
#return first_customer_record, sort_keys=True, indent=4))    
#print(json.dumps({'4': 5, '6': 7}, sort_keys=True, indent=4))


# In[6]:


line = getFirstLineJSON(grocery_json_raw)


# In[7]:


line


# # Parse sentenses and output lines to file

# In[8]:


import spacy
import itertools as it

nlp = spacy.load('en')


# In[19]:


get_ipython().run_cell_magic('time', '', "parsed_review = nlp(line['reviewText'])")


# In[20]:


parsed_review


# In[21]:


#line=json.loads(line)


# In[22]:


#line= line.replace('\n','')


# In[23]:


line['reviewerID']


# In[ ]:



    


# In[14]:


#dir(json.dumps)


# In[15]:


import json

class Object:
    def toJSON(self):
        return json.dumps(self, 
                          default=lambda o: o.__dict__, 
                          sort_keys=True, indent=4)


# In[25]:


import json
import sys
dataset={}
#line = json.load(sample)
        #line = json.load(line)
        #print(line["asin"])
        #sample_review=line[0]
        #line = eval(line)
        #print(sample)
    #parsed_review = nlp(line["reviewText"])
i=0


for line in open(grocery_json_raw).readlines():
    line=json.loads(line)
    parsed_review=nlp(line['reviewText'])
    j=0
    file = open(grocery_json_flattened, 'a') 
    for num, sentence in enumerate(parsed_review.sents):
        
        #print('Sentence {}:'.format(num + 1))
        #print(sentence)
        #print('')
        data = Object()
        data.asin = line['asin']
        data.helpful = line['helpful']
        data.overall = line['overall']
        data.reviewTime = line['reviewTime']
        data.reviewerID = line['reviewerID']
        #if not line["reviewerName"]:
        #    pass
        #else:
        #    data.reviewerName = line["reviewerName"]
            
        data.summary = line['summary']
        data.unixReviewTime = line['unixReviewTime']
        data.sentence_Number =num
        data.sentence = sentence.text
        
        file.write(data.toJSON())
    file.close()



# In[ ]:


read_file = open(grocery_json_flattened, 'r')


# In[ ]:


from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct


# In[ ]:


j = dumps(data, cls=PythonObjectEncoder)

loads(j, object_hook=as_python_object)


# In[ ]:


feeds={}
with open(grocery_json_flattened, mode='w', encoding='utf-8') as feedsjson:
    #entry = {'name': args.name, 'url': args.url}
    #feeds.append(entry)
    feeds.append(data)
    json.dump(feeds, feedsjson)


# In[ ]:


#with open(grocery_json_flattened, mode='w+', encoding='utf-8') as f:
#    json.dump({}, f)


# In[ ]:


import json

a_dict = {'new_key': 'new_value'}

#with open(grocery_json_flattened) as f:
#    data = json.load(f)

#data.update(a_dict)
#data.update(a_dict)
with open(grocery_json_flattened, 'a+') as f:
    json.dump(a_dict, f)


# In[29]:


import pandas as pd


# In[194]:


sample=pd.read_json(grocery_json_sample, lines=True)


# In[195]:


len(sample.columns)


# In[196]:


meta = pd.read_json(metadata_json_raw, lines=True)


# In[43]:


len(meta.columns)


# In[197]:


meta.head(1)


# In[38]:


results = pd.merge(sample, meta, how='inner', on=['asin', 'asin'])


# In[52]:


results.iloc[1:2,:]


# # Sample JSON
# {
#     "reviewerID":"A1VEELTKS8NLZB",<br/>
#     "asin":"616719923X",<br/>
#     "reviewerName":"Amazon Customer",<br/>
#     "helpful":[<br/>
#         0,<br/>
#         0<br/>
#     ],<br/>
#     "reviewText":"Just another flavor of Kit Kat but the taste is unique and a bit different.  The only thing that is bothersome is the price.  I thought it was a bit expensive....",<br/>
#     "overall":4.0,<br/>
#     "summary":"Good Taste",<br/>
#     "unixReviewTime":1370044800,<br/>
#     "reviewTime":"06 1, 2013"<br/>
# }

# In[123]:


import spacy

nlp = spacy.load('en')

GOOD_ENTS = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC',
             'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE']

def entities(sent):
    """
    Given a sentence, returns entities, represented at 2-tuples
    with entity text (str) and Spacy entity type label (str)
    """
    #doc = nlp(sent)
    doc=sent
    for ent in doc.ents:
        #  filter out non social entities
        if ent.label_ in GOOD_ENTS:
            return ent.text, ent.label_
        else:
            pass


# In[137]:


import itertools

def pairs(sent):
    
    """
    Given a document (list of paras, which is a list of sents,
    which is a list of token,tag tuples), return a list of all
    entity pairs.
    """
    candidates = [
        entities(' '.join(word for word, tag in sent))
        #for para in doc for sent in para
        #for idx, sent in enumerate(doc)
    ]

    doc_entities = [
        entity for entity in candidates if entity is not None
    ]

    return list(itertools.permutations(set(doc_entities), 2))


# asin                                                     616719923X
# helpful                                                      [0, 1]
# overall                                                           3
# reviewText        I bought this on impulse and it comes from Jap...
# reviewTime                                              05 19, 2014
# reviewerID                                           A14R9XMZVJ6INB
# reviewerName                                                amf0001
# summary           3.5 stars,  sadly not as wonderful as I had hoped
# unixReviewTime                                           1400457600
# brand                                                           NaN
# categories                               [[Grocery & Gourmet Food]]
# description       Green Tea Flavor Kit Kat have quickly become t...
# imUrl             http://ecx.images-amazon.com/images/I/51LdEao6...
# price                                                           NaN
# related           {'also_bought': ['B00FD63L5W', 'B0047YG5UY', '...
# salesRank                         {'Grocery & Gourmet Food': 37305}
# title             Japanese Kit Kat Maccha Green Tea Flavor (5 Ba...

# In[354]:


import matplotlib.pyplot as plt
import networkx as nx

from __future__ import unicode_literals, print_function

import plac
import spacy
import sys

title_attributes = set()
reviewerID_ttributes = set()
reviewText_attributes = set()
        
def appendnode(G,node, attribute):
    #node = row['asin']
    #attribute = row['asin'] #product ID
    if not G.has_node(node):
        G.add_node(node, attribute=attribute)
    return G
        
        
def graph(corpus):
    """
    Each document is considered to be unique.  In my case, each review is considered to be unique.
    
    fileid is equivalent to "asin & reviewerID" and uniquely identifies the review, 
    linke reviewerID
    product is asin
    """
 
    G = nx.Graph(name="Amazon Product Review Graph") 
    i=0
    try:
        
        #title_attributes = set()
        #reviewerID_ttributes = set()
        #reviewText_attributes = set()

        # code here adding nodes

        


        #node = row['asin']
        #attribute = row['asin'] #product ID
        #if attribute not in myset:
        #    G.add_node(node, attr=attribute)
        #    title_attributes.add(attribute)
       


        # Create category, feed, and document nodes
        #G.add_nodes_from(corpus.categories(), type='category')
        #G.add_nodes_from([feed['title'] for feed in corpus.feeds()], type='feed')
        #G.add_nodes_from(corpus.fileids(), type='document')
        j=0
        entity=[['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC',
                 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'MONEY', 
                 'QUANTITY']]
        
        G=appendnode(G,'PERSON','PERSON')
        G=appendnode(G,'NORP', 'NORP')
        G=appendnode(G,'FACILITY', 'FACILITY')
        G=appendnode(G,'ORG', 'ORG')
        G=appendnode(G,'GPE', 'GPE')
        G=appendnode(G,'LOC', 'LOC')
        G=appendnode(G,'PRODUCT', 'PRODUCT')
        G=appendnode(G,'EVENT', 'EVENT')
        G=appendnode(G,'WORK_OF_ART', 'WORK_OF_ART')
        G=appendnode(G,'LANGUAGE', 'LANGUAGE')
        G=appendnode(G,'MONEY', 'MONEY')
        G=appendnode(G,'QUANTITY', 'QUANTITY')
                            
        
        for index, row in corpus.iterrows():
                 
            # Product 
            node = row['asin']
            attribute = row['asin'] #product ID
            G=appendnode(G,node, attribute)
            
            # Reviewer
            node = row['reviewerID']
            attribute = row['reviewerID']
            G=appendnode(G,node, attribute)
            
            # Review
            node = row['reviewText']
            attribute = row['reviewText'] 
            G=appendnode(G,node, attribute)
            
            # Create feed-category edges
            
            # Create 
            if not (G.has_edge(row['reviewerID'],row['reviewText']) and
                    G.has_edge(row['asin'],row['reviewText'])):
                
                G.add_edge(row['reviewerID'], row['reviewText']) 
                G.add_edge(row['asin'], row['reviewText'])
                    
            j+=1
            
            #Extract and assign Entities
            model='en_core_web_sm'
            nlp = spacy.load(model)
            doc = nlp(row['reviewText'] )
            
            relations = extract_entity_relations(doc)
            print(relations)
            for r1, r2 in relations:
                print('{:<10}\t{}\t{}'.format(r1.text, r2.ent_type_, r2.text))
                if not G.has_edge(row['asin'], r2.ent_type_):
                    G.add_edge(row['asin'], r2.ent_type_)
                else:
                    G.add_edge(row['asin'], r2.ent_type_, weight=1)
        return G
    
    
        for index, row in corpus.iterrows():
            print(i)
            i+=1
            if i==1:
                break
            # Create an undirected graph
            #G = nx.Graph(name="Amazon Product Review Graph")

            # Create category, feed, and document nodes
            #G.add_nodes_from(row['asin'], type='product')
            #G.add_nodes_from(row['reviewerID'], type='reviewer')
            #G.add_nodes_from(row['summary'], type='document')

        # Create feed-category edges
        # Create Reviewer-PRODUCT edges
            G.add_edges_from([
                (row['reviewerID'], row['asin'])
            ])

        # Create document-category edges
        # Create document-product edges
            G.add_edges_from([
                (row['summary'], row['title'])
            ])
    
        # Add edges for each document-entity relationship
        # and between pairs of related entities

#@plac.annotations(
#    model=("Model to load (needs parser and NER)", "positional", None, str))
            model='en_core_web_sm'
            nlp = spacy.load(model)
            doc = nlp(row['reviewText'])
            #for num, sentence in enumerate(parsed_review.sents):
            #for text in TEXTS:
                #doc = nlp(text)
            relations = extract_entity_relations(doc)
        
            for r1, r2 in relations:
                print('{:<10}\t{}\t{}'.format(r1.text, r2.ent_type_, r2.text))
                G.add_edge(row['summary'], r2.ent_type_)
                G.add_edge(row['reviewerID'], r2.ent_type_)
                G.add_edge(row['reviewerID'], r2.text )

                # Now add edges between entity pairs with a weight
                # of 1 for every document they co-appear in

                if (r1.text, r2.text) in G.edges():
                    G.edges[(r1.text, r2.text)]['weight'] += 1
                else:
                    G.add_edge(r1.text, r2.text, weight=1)
                
                if (row['reviewText'],r2.ent_type_) in G.edges():
                    G.edges[(row['reviewText'], r2.ent_type_)]['weight'] += 1
                else:    
                    G.add_edge(row['reviewText'], r2.ent_type_, weight=1)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
        
    #return G        
                      
            
              
                


# In[355]:


def extract_entity_relations(doc):
    # merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    for span in spans:
        span.merge()

    relations = []
    #GOOD_ENTS = ['PRODUCT']
    GOOD_ENTS = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC',
                 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'MONEY', 
                 'QUANTITY']
    for entity in filter(lambda w: w.ent_type_ in GOOD_ENTS, doc):

        if entity.dep_ in ('attr', 'dobj'):
            subject = [w for w in entity.head.lefts if w.dep_ == 'nsubj']
            if subject:
                subject = subject[0]
                relations.append((subject, entity))
                #relations.append((entity))
        elif entity.dep_ == 'pobj' and entity.head.dep_ == 'prep':
            relations.append((entity.head.head, entity))
            #relations.append((entity))
            
    return relations  


# In[356]:


G = graph(results)


# In[358]:


len(G)


# In[359]:


#Centrality
import heapq
from operator import itemgetter


def nbest_centrality(G, metric, n=10, attr="centrality", **kwargs):
    # Compute the centrality scores for each vertex
    scores = metric(G, **kwargs)

    # Set the score as a property on each node
    nx.set_node_attributes(G, name=attr, values=scores)

    # Find the top n scores and print them along with their index
    topn = heapq.nlargest(n, scores.items(), key=itemgetter(1))
    for idx, item in enumerate(topn):
        print("{}. {}: {:0.4f}".format(idx + 1, *item))

    return G


# In[63]:


results.to_pickle(pickledResults)


# In[64]:


results=pd.read_pickle(pickledResults)


# In[65]:


type(results)


# In[75]:


results.loc['title']


# In[185]:


results.loc[1,'title']


# In[360]:


nbest_centrality(G, nx.degree_centrality, 10, "degree")


# In[363]:


G.info()


# In[176]:





# # Named Entity Recognition<br/>
# Type	Description<br/>
# PERSON	People, including fictional.<br/>
# NORP	Nationalities or religious or political groups.<br/>
# FACILITY	Buildings, airports, highways, bridges, etc.<br/>
# ORG	Companies, agencies, institutions, etc.<br/>
# GPE	Countries, cities, states.<br/>
# LOC	Non-GPE locations, mountain ranges, bodies of water.<br/>
# PRODUCT	Objects, vehicles, foods, etc. (Not services.)<br/>
# EVENT	Named hurricanes, battles, wars, sports events, etc.<br/>
# WORK_OF_ART	Titles of books, songs, etc.<br/>
# LAW	Named documents made into laws.<br/>
# LANGUAGE	Any named language.<br/>
# DATE	Absolute or relative dates or periods.<br/>
# TIME	Times smaller than a day.<br/>
# PERCENT	Percentage, including "%".<br/>
# MONEY	Monetary values, including unit.<br/>
# QUANTITY	Measurements, as of weight or distance.<br/>
# ORDINAL	"first", "second", etc.<br/>
# CARDINAL	Numerals that do not fall under another type.<br/>

# In[364]:


print("Betweenness centrality")
nbest_centrality(
    G, nx.betweenness_centrality, 10, "betweenness", normalized=True
)


# In[376]:


from multiprocessing import Pool
import itertools


def partitions(nodes, n):
	"Partitions the nodes into n subsets"
	nodes_iter = iter(nodes)
	while True:
		partition = tuple(itertools.islice(nodes_iter,n))
		if not partition:
			return
		yield partition

# To begin the parallel computation, we initialize a Pool object with the
# number of available processors on our hardware. We then partition the
# network based on the size of the Pool object (the size is equal to the 
# number of available processors). 
def between_parallel(G, processes = None):
	p = Pool(processes=processes)
	part_generator = 4*len(p._pool)
	node_partitions = list(partitions(G.nodes(), int(len(G)/part_generator)))
	num_partitions = len(node_partitions)

    #Next, we pass each processor a copy of the entire network and 
    #compute #the betweenness centrality for each vertex assigned to the 
    #processor.

	bet_map = p.map(btwn_pool,
					zip([G]*num_partitions,
						[True]*num_partitions,
						[None]*num_partitions,
						node_partitions))

    #Finally, we collect the betweenness centrality calculations from each 
    #pool and aggregate them together to compute the overall betweenness 
    #centrality score for each vertex in the network.

	bt_c = bet_map[0]
	for bt in bet_map[1:]:
		for n in bt:
			bt_c[n] += bt[n]
	return bt_c


# In[ ]:


import community

parts = community.best_partition(G)
values = [parts.get(node) for node in G.nodes()]



# In[402]:


from operator import itemgetter
import networkx as nx
import matplotlib.pyplot as plt


#if __name__ == '__main__':
    # Create a BA model graph
n=1000
m=2

    #G=nx.generators.barabasi_albert_graph(n,m)
    # find node with largest degree
#node_and_degree=G.degree()
#(largest_hub,degree)=sorted(node_and_degree.items(),key=itemgetter(1))[-1]
    # Create ego graph of main hub
#hub_ego = nx.ego_graph(G, "ORG")
#hub_ego=nx.ego_graph(G,largest_hub)
    # Draw graph
#pos=nx.spring_layout(hub_ego)
#nx.draw(hub_ego,pos,node_color='b',node_size=50,with_labels=False)
    # Draw ego as large and red
#nx.draw_networkx_nodes(hub_ego,pos,nodelist=[largest_hub],node_size=300,node_color='r')
#plt.savefig('ego_graph.png')
#plt.show()


# In[378]:


print(nx.info(G))


# In[379]:


spring_pos = nx.spring_layout(G)


# In[382]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[398]:


plt.axis("off")
nx.draw_networkx(G, pos = spring_pos,with_labels=False, node_size=2)


# In[384]:


#plt.savefig('ego_graph.png')


# In[395]:


plt.show(block=False)


# In[399]:


plt.savefig("ego_graph.png", format="PNG")


# In[401]:


plt.axis("off")
nx.draw_networkx(G, pos = spring_pos,with_labels=False, node_size=2)

