### preprocess
#tweet-preprocessor: https://github.com/s/preprocessor
import numba
import numpy.random
import preprocessor as p
import time
import pandas as pd
import numpy as np
import re
import pysbd
from sentence_transformers import SentenceTransformer
import pickle
# PANDAS WILL DROP ROWS SILENTLY!!!  We need to set quoting=csv.QUOTE_NONE
import bz2
import csv
from tqdm import tqdm
from umap import UMAP, ParametricUMAP
#from hdbscan import HDBSCAN
import fast_hdbscan
import os
import os

from umap.parametric_umap import load_ParametricUMAP

homedir = os.getenv("HOME")
os.chdir(homedir+"/nzTwitterCovid/data/bert")

title = "nzTwitterAuckland"
uniques_file = 'uniques_' + title + '.tsv'
all_file = 'all_sentences_' + title + '.tsv'
embeddings_file = 'embedding_' + title + '.pbz2'
umap_file = 'umap_' + title + '.tsv'

the_data = pd.read_csv("./tweetsOnlyAuckland.tsv",sep="\t", header=None, quoting=csv.QUOTE_NONE)
the_data = the_data.rename(columns={0: "DATE", 1: "LOCATION", 2: "TEXT", 3: "AUCKLAND"})

seg = pysbd.Segmenter(language="en", clean=False)
# in my experience HDBSCAN separates hashtags into separate clusters anyway
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.HASHTAG)
uniqueSentences = {}
allSentences = []
indices = []
# I'm sure there are way faster ways to do this, but whatever.
# I'm also forced to str() the text, as some fields are being returned as floats by python's Pandas, I assume.
for index in tqdm(the_data.index):
	text = str(the_data['TEXT'][index]).strip()
	# remove unicode characters
	text = re.sub(r'<U\+[a-zA-Z0-9]{4,8}>', '', text)
	# remove unicode characters
	text = re.sub(r'\\[0-9]{3}', '', text)
	text = p.clean(text)
	sentences = seg.segment(text)
	if len(sentences) == 0:
		uniqueSentences.update({"": index})
		allSentences.append("")
		indices.append(index)
	else:
		for sentence in sentences:
			sentence = str(sentence).strip()
			uniqueSentences.update({sentence: index})
			allSentences.append(sentence)
			indices.append(index)




def send2UMAP(embedding_file,umap_file):
	with bz2.BZ2File(embedding_file, 'rb') as pkl:
		embedding_model = pickle.load(pkl)
	@numba.njit()
	def set_random_njit():
		np.random.seed(42)
	umap_model = UMAP(n_components=5, random_state=42, min_dist=0.0, spread=0.5, metric='cosine', verbose=True)
	set_random_njit()
	umap_model.fit(embedding_model)
	umap_embedding = umap_model.embedding_
	pd.DataFrame(umap_embedding).to_csv(umap_file, sep="\t", index=False, header=False)

def send2fast_HDBSCAN(umap_file,cluster_file,min_cluster_percentage,min_samples):
	import csv
	umapped = pd.read_csv(umap_file,keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
	dims = np.array(umapped)
	min_cluster_size = round(len(dims)*min_cluster_percentage)
	print("Min cluster size: " + str(min_cluster_size))
	hdbscan_model = fast_hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size, cluster_selection_method='leaf', prediction_data=False)
	hdbscan_model.fit(dims)
	clusters = hdbscan_model.labels_
	#probs = hdbscan_model.probabilities_
	uniques = np.unique(np.array(clusters), return_counts=True)
	print("Distribution: " + str(uniques))
	print("Uniques: " + str(len(uniques[0])))
	print("-1 %: " + str(uniques[1][0]/len(clusters)))
	# convert array into dataframe
	#pd.DataFrame({'clusters': clusters, 'probs': probs}).to_csv(cluster_file, index=False, sep='\t',quoting=csv.QUOTE_NONE)
	pd.DataFrame({'clusters': clusters}).to_csv(cluster_file, index=False, sep='\t',quoting=csv.QUOTE_NONE)

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



send2UMAP(embeddings_file,umap_file)

min_cluster_percentage = 0.002
min_samples = 200
hdbscan_file = "hdbscan_" + title + "_"+str(min_cluster_percentage)+"_mcp_"+str(min_samples)+"_ms.tsv"
send2fast_HDBSCAN(umap_file,hdbscan_file,min_cluster_percentage,min_samples)
model_BERTopic = send2BERTopic(uniques_file,umap_file,hdbscan_file)
results2browser(resultsFromBERT2HDBSCANtopics(model_BERTopic,hdbscan_file))
results2csv(resultsFromBERT2HDBSCANtopics(model_BERTopic,hdbscan_file),"results_"+hdbscan_file)
