import numba
import numpy.random
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import bz2
import csv
from tqdm import tqdm
import os
from bertopic import BERTopic

homedir = os.getenv("HOME")
os.chdir(homedir+"/nzTwitterCovid/data/bert")


def send2BERTopic(uniques, embeddings, clusters):
	from bertopic import BERTopic
	from sklearn.feature_extraction.text import CountVectorizer
	import spacy
	# spacy.cli.download('en_core_web_sm')
	spacy.load('en_core_web_sm')
	from bertopic.representation import PartOfSpeech
	from bertopic.vectorizers import ClassTfidfTransformer
	from sentence_transformers import SentenceTransformer
	from bertopic.dimensionality import BaseDimensionalityReduction
	from bertopic.cluster import BaseCluster
	from bertopic.representation import KeyBERTInspired
	from bertopic.vectorizers import OnlineCountVectorizer
	from bertopic.representation import MaximalMarginalRelevance
	uniques = pd.read_csv(uniques, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)[0]
	embeddings = np.array(pd.read_csv(embeddings, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE))
	clusters = pd.read_csv(clusters, keep_default_na=False, sep='\t', quoting=csv.QUOTE_NONE)['clusters']
	representation_models = [MaximalMarginalRelevance(diversity=1),PartOfSpeech("en_core_web_sm")]
	vectorizer_model = OnlineCountVectorizer(stop_words="english")
	ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
	embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
	empty_reduction_model = BaseDimensionalityReduction()
	empty_cluster_model = BaseCluster()
	# Fit BERTopic without constructing embeddings
	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=empty_reduction_model,
		hdbscan_model=empty_cluster_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_models,
		verbose=True
	).fit(uniques, embeddings=embeddings, y=clusters)
	print("Customizing representative docs.")
	#representative_docs is screwed up for a number of reasons (pysbd doesn't properly parse " '; it tends to give long answers, etc.) let's fix that:
	revised_docs = pd.DataFrame({'Document': uniques, 'Topic': topic_model.topics_})
	revised_docs = revised_docs[~revised_docs.Document.str.contains("\"")]
	revised_docs = revised_docs[~revised_docs.Document.str.contains("\'")]
	revised_docs = revised_docs[revised_docs.Document.str.len() > 15]
	revised_docs = revised_docs[revised_docs.Document.str.len() < 225]
	revised_docs = revised_docs.reset_index(drop=True)
	# sample all of the remaining data and give 10 sentences.
	tmprepr_docs_mappings, tmprepr_docs, tmprepr_docs_indices, tmprepr_docs_ids = topic_model._extract_representative_docs(c_tf_idf=topic_model.c_tf_idf_,
	                                                      documents=revised_docs,
	                                                      topics=topic_model.topic_representations_,
	                                                      nr_samples=round(revised_docs.shape[0]),
	                                                      nr_repr_docs=10)
	topic_model.representative_docs_= tmprepr_docs_mappings
	return topic_model


def resultsFromBERT2HDBSCANtopics(topic_model,hdbscan_file):
	import re
	map_BERTandHDBSCAN_topic_numbers = pd.DataFrame({'BERT': np.array(topic_model.topics_),
	                                                 'HDBSCAN': np.array(pd.read_csv(hdbscan_file,
	                                                                                 keep_default_na=False, sep='\t',
	                                                                                 quoting=csv.QUOTE_NONE)[
		                                                                     'clusters'])}).drop_duplicates().set_index(
	'BERT')['HDBSCAN'].to_dict()
	results = topic_model.get_topic_info()
	# let's repopulate representative docs...
	for i in range(len(results['Topic'])):
		results['Topic'][i] = map_BERTandHDBSCAN_topic_numbers.get(results['Topic'][i])
		results['Name'][i] = re.sub("^[0-9]*_",str(results['Topic'][i])+"_",results['Name'][i])
	return results



def results2browser(results):
	import os, tempfile
	resultsHTML = pd.DataFrame(results).to_html(classes=["table-bordered", "table-striped", "table-hover"])
	tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".html")
	try:
		print(tmp.name)
		tmp.write(resultsHTML.encode('ascii'))
		os.system("firefox " + tmp.name)
		time.sleep(2)
		tmp.close()
	finally:
		os.unlink(tmp.name)
	return

def results2csv(results,results_file):
	results.to_csv(results_file,index=False, sep='\t')


# I've had better success with merging via forced ctfidf comparison, than merging two different dimensionally reduced embeddings via cosine similarity (perhaps because of reduction process?)
def topic_embeddings_byctfidf(themodel):
	topic_list = list(themodel.topic_representations_.keys())
	topic_list.sort()
	# Only extract top n words
	n = len(themodel.topic_representations_[topic_list[0]])
	if themodel.top_n_words < n:
	    n = themodel.top_n_words
	# Extract embeddings for all words in all topics
	topic_words = [themodel.get_topic(topic) for topic in topic_list]
	topic_words = [word[0] for topic in topic_words for word in topic]
	word_embeddings = themodel.embedding_model.embed_words(
	    words=topic_words,
	    verbose=True
	)
	print(word_embeddings)
	# Take the weighted average of word embeddings in a topic based on their c-TF-IDF value
	# The embeddings var is a single numpy matrix and therefore slicing is necessary to
	# access the words per topic
	topic_embeddings = []
	for i, topic in enumerate(topic_list):
	    word_importance = [val[1] for val in themodel.get_topic(topic)]
	    if sum(word_importance) == 0:
	        word_importance = [1 for _ in range(len(themodel.get_topic(topic)))]
	    topic_embedding = np.average(word_embeddings[i * n: n + (i * n)], weights=word_importance, axis=0)
	    topic_embeddings.append(topic_embedding)
	return np.array(topic_embeddings)




modelAuckland_BERTopic = send2BERTopic("./uniques_nzTwitterAuckland.tsv","./umap_nzTwitterAuckland.tsv","./hdbscan_nzTwitterAuckland_0.002_mcp_200_ms.tsv")
modelOutsideAuckland_BERTopic = send2BERTopic("./uniques_nzTwitterOutsideAuckland.tsv","./umap_nzTwitterOutsideAuckland.tsv","./hdbscan_nzTwitterOutsideAuckland_0.002_mcp_200_ms.tsv")

modelAuckland_BERTopic.topic_embeddings_ = topic_embeddings_byctfidf(modelAuckland_BERTopic)
modelOutsideAuckland_BERTopic.topic_embeddings_ = topic_embeddings_byctfidf(modelOutsideAuckland_BERTopic)

modelMerged = BERTopic.merge_models([modelAuckland_BERTopic,modelOutsideAuckland_BERTopic])
results2csv(modelMerged.get_topic_info(),"./results_merged.tsv")
results2csv(pd.DataFrame({'clusters': modelMerged.topics_}),"./hdbscan_merged.tsv")