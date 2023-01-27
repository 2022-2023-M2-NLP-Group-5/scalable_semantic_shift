import pickle
import pandas as pd
import ot
import argparse
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from collections import Counter
from scipy.stats import entropy
from collections import defaultdict
import numpy as np
import sys
from scipy.stats import wasserstein_distance as wd
from scipy.spatial import distance

def extract_frequent_embeddings(slice_embeddings, slice, postfix="_text", topn=3):
    slice_text = slice + postfix
    senses = slice_embeddings[slice_text]
    occurrences = dict()
    for sense in senses:
        occurrences[sense] = len(senses[sense])
    occurrences_sorted = {k: v for k, v in sorted(occurrences.items(), 
        key=lambda item: item[1])}
    result_embeddings = []
    for key in occurrences_sorted.keys():
        result_embeddings.append(slice_embeddings[slice][key][0])
    return result_embeddings[:topn]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure semantic shift')
    parser.add_argument("--method", default='WD', const='all', nargs='?',
                        help="A method for calculating distance", choices=['WD', 'COS'])
    parser.add_argument("--corpus_slices",
                        default='1910;1950',
                        type=str,
                        help="Time slices names separated by ';'.")
    parser.add_argument("--get_additional_info", action="store_true", help='Whether the cluster labels and sentences, required for interpretation, are saved or not.')
    parser.add_argument('--results_dir_path', type=str, default='results_coha', help='Path to the folder to save the results.')
    parser.add_argument('--embeddings_path', type=str, default='data/embeddings/', help='Path to the embeddings pickle file.')
    parser.add_argument('--define_words_to_interpret', type=str, default='', help='Define a set of words separated by ";" for interpretation if you do not wish to save data for all words.')
    parser.add_argument('--random_state', type=int, default=420, help='Choose a random state for reproducibility of clustering.')
    parser.add_argument('--cluster_size_threshold', type=int, default=7, help='Clusters smaller than a threshold will be merged or deleted.')
    parser.add_argument('--top_n', type=int, default=1, help='Decide on top n most frequent occcurrences')
    args = parser.parse_args()
    random_state = args.random_state
    threshold = args.cluster_size_threshold
    get_additional_info = args.get_additional_info
    print(args.embeddings_path)
    embeddings_path = args.embeddings_path
    corpus_slices = args.corpus_slices.split(';')
    top_n = args.top_n
    print("Corpus slices:", corpus_slices)

    methods = ['WD', 'COS']
    if args.method not in methods:
        print("Method not valid, valid choices are: ", ", ".join(methods))
        sys.exit()

    if args.method == "WD":
        method = wd
    else: 
        method = distance.cosine

    print("Loading ", " and ".join(corpus_slices))
    embeddings = dict()
    c2ss = dict()
    for slice in corpus_slices:
        file_path = embeddings_path + slice + ".pickle"
        try:
            bemb , c2s = pickle.load(open(file_path, 'rb'))
        except:
            bemb = pickle.load(open(file_path, 'rb'))
            c2s = None
        embeddings[slice] = bemb
        c2ss[slice] = c2s
    
    if len(args.define_words_to_interpret) > 0:
        target_words = args.define_words_to_interpret.split(';')
    else:
        vocabs = set()
        for key in embeddings.keys():
            bemb = embeddings[key]
            vocabs = vocabs.union(bemb.keys())
        target_words = list(vocabs)

    # For merging the embeddings and the count2sents
    bert_embeddings = dict()
    for word in target_words:
        for slice in embeddings.keys():
            slice_data = embeddings[slice]
            try:
                data_partitions = slice_data[word].keys()
            except KeyError:
                continue
            try:
                _ = bert_embeddings[word]
            except KeyError:
                bert_embeddings[word] = dict()
            for partition in data_partitions:
                    bert_embeddings[word][partition] = slice_data[word][partition]
    # For merging the count2sents
    count2sents = dict()
    for slice in c2ss.keys():
        try:
            _ = c2ss[slice]
        except KeyError:
            count2sents[slice] = dict()
        count2sents[slice] = c2ss[slice][slice]


    if get_additional_info and len(target_words) > 10:
        print('Define a list of words to interpret with less than 100 words or set "get_additional_info" flag to False')
        sys.exit()

    records = []
    
    for word in target_words:
        w_embeddings = bert_embeddings[word]
        
        for slice in corpus_slices:
            slice_vocab = set(embeddings[slice].keys())
            try:
                slice_vocab.remove(word)
            except KeyError:
                print("{0} is not in the {1} slice.".format(word, slice))
                continue
            local_record = dict()
            local_record["word"] = word
            local_record["slice"] = slice
            senses = extract_frequent_embeddings(w_embeddings, slice, topn=top_n)
            for i, sense in enumerate(senses):
                distances = []
                for vocab in slice_vocab:
                    try:
                        vocab_embeddings = extract_frequent_embeddings(embeddings[slice][vocab], slice, topn=1)
                        distances.append((vocab, method(sense, vocab_embeddings[0])))

                    except:
                        continue
                distances.sort(key=lambda a: a[1])
                relatives = [k for k,v in distances[:threshold]]
                relatives_distances = [str(v) for k,v in distances[:threshold]]
                local_record["cluster"] = ";".join(relatives)
                local_record["distance"] = ";".join(relatives_distances)
                local_record["sense"] = i+1
                records.append(local_record)
    pd.DataFrame.from_dict(records).to_csv("csv/{}_{}.csv".format(word,"-".join(corpus_slices)))
    # Extract the semantic changes between two words
    if len(target_words==2):
        pair_distances = dict()
        for word in target_words:
            w_embeddings = bert_embeddings[word]
            element_embeddings = []
            for slice in corpus_slices:
                slice_vocab = set(embeddings[slice].keys())
                try:
                    slice_vocab.remove(word)
                except KeyError:
                    print("{0} is not in the {1} slice.".format(word, slice))
                    continue
                local_record = dict()
                local_record["word"] = word
                local_record["slice"] = slice
                senses = extract_frequent_embeddings(w_embeddings, slice, topn=1)
                element_embeddings.append(senses[0])
            pair_distances[word] = element_embeddings
