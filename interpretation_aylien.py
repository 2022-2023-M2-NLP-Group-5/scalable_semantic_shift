from sklearn.feature_extraction.text import CountVectorizer
import pickle
import argparse
import spacy

spacy_nlp = spacy.load('en_core_web_sm')
# python -m spacy download en
nlp = spacy.load('en')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.pyplot import subplots
from collections import defaultdict
from lemmagen3 import Lemmatizer



def get_clusters_sent(target, method, corpus_slices_type, threshold_size_cluster):

    labels = "aylien_most_changed_words_results/" + method + "_labels_english_fine_tuned_averaged.pkl"
    sentences = "aylien_most_changed_words_results/sents_english_fine_tuned_averaged.pkl"
    corpus_slices = []
    if corpus_slices_type == 'months':
        corpus_slices = ['january', 'february', 'march', 'april']
    elif corpus_slices_type == 'sources':
        corpus_slices = ['fox', 'cnn']
    else:
        print("corpus_slices_type argument not recognised")

    labels = pickle.load(open(labels, 'rb'))
    print(labels.keys())
    sentences = pickle.load(open(sentences, 'rb'))

    cluster_to_sentence = defaultdict(lambda: defaultdict(list))
    for cs in corpus_slices:
        for label, sent in zip(labels[target][cs], sentences[target][cs]):
            cluster_to_sentence[label][cs].append(sent)

    counts = {cs: Counter(labels[target][cs]) for cs in corpus_slices}
    common_counts = []
    all_labels = []
    for slice, c in counts.items():
        slice_labels = [x[0] for x in c.items()]
        all_labels.extend(slice_labels)
    all_labels = set(all_labels)
    all_counts = []
    for l in all_labels:
        all_count = 0
        for slice in corpus_slices:
            count = counts[slice][l]
            all_count += count
        all_counts.append((l, all_count))

    sorted_counts = sorted(all_counts, key=lambda x: x[1], reverse=True)
    sentences = []
    labels = []
    categs = []

    for label, count in sorted_counts:
        print("\n================================")
        print("Cluster label: ", label, " - Cluster size: ", count)
        if count > threshold_size_cluster:
            for cs in corpus_slices:
                print("Corpus slice: ", cs, " - Num appearances: ", counts[cs].get(label, 0))
                for s in cluster_to_sentence[label][cs]:
                    sent_clean = s.replace("[CLS]", "").replace("[SEP]", "").strip()
                    # sentences_dict[label][cs].append(sent_clean)
                    sentences.append(sent_clean)
                    labels.append(label)
                    categs.append(cs)
        else:
            print("Cluster", label, "is too small - deleted!")

    sent_df = pd.DataFrame(list(zip(sentences, labels, categs)), columns=['sentence', 'label', 'categ'])

    return sent_df


def output_distrib(data, word, order):
    k = data['label'].unique()
    distrib = data.groupby(['categ', "label"]).size().reset_index(name="count")
    pivot_distrib = distrib.pivot(index='categ', columns='label', values='count')
    pivot_distrib_norm = pivot_distrib.div(pivot_distrib.sum(axis=1), axis=0)
    fig, ax = subplots()
    axs = pivot_distrib_norm.loc[order].plot.bar(stacked=True, title="Word: \"" + word + '\"', colormap='Spectral',
                                                 ax=ax)
    ax.legend(title='Cluster', loc='upper right')
    x_axis = axs.axes.get_xaxis()
    x_axis.label.set_visible(False)
    plt.xticks(rotation=0)
    plt.savefig("distrib.png", dpi=(300))


    return pivot_distrib_norm


def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


def extract_keywords(target_word, word_clustered_data, max_df, topn):
    lemmatizer = Lemmatizer('en')
    # get groups of sentences for each cluster
    l_sent_clust_dict = defaultdict(list)
    sent_clust_dict = defaultdict(list)
    for i, row in word_clustered_data.iterrows():
        l_sent_clust_dict[row['label']].append(row['sentence'])

    for label, sents in l_sent_clust_dict.items():
        sent_clust_dict[label] = " ".join(sents)

    stop1 = list(spacy.lang.en.stop_words.STOP_WORDS)
    stop2 = stopwords.words('english')
    stop = set(stop1 + stop2)

    labels, clusters = list(sent_clust_dict.keys()), list(sent_clust_dict.values())

    # print(list(cv.vocabulary_.keys())[:10])
    tfidf_transformer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=(1,2), max_df=max_df, stop_words=stop, max_features=10000)
    tfidf_transformer.fit(clusters)
    feature_names = tfidf_transformer.get_feature_names()

    keyword_clusters = {}
    for label, cluster in zip(labels, clusters):
        # generate tf-idf
        tf_idf_vector = tfidf_transformer.transform([cluster])
        # sort the tf-idf vectors by descending order of scores
        tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        # extract only the top n
        keywords = extract_topn_from_vector(feature_names, sorted_items, topn*5)
        keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        keywords = [x[0] for x in keywords]
        #filter unigrams that appear in bigrams and remove duplicates
        all_bigrams = " ".join([kw for kw in keywords if len(kw.split()) == 2])
        already_in = set()
        filtered_keywords = []
        for kw in keywords:
            if len(kw.split()) == 1 and kw in all_bigrams:
                continue
            else:
                if len(kw.split()) == 1:
                    kw = lemmatizer.lemmatize(kw)
                if kw not in already_in and kw != target_word:
                    filtered_keywords.append(kw)
                    already_in.add(kw)

        keyword_clusters[label] = filtered_keywords[:topn]

    return keyword_clusters


def full_analysis(word, max_df=0.8, topn=15, method, corpus_slices, threshold_size_cluster):
    clusters_sents_df = get_clusters_sent(word, method, corpus_slices, threshold_size_cluster)
    pivot_distrib = output_distrib(clusters_sents_df, word, corpus_slices)
    keyword_clusters = extract_keywords(word, clusters_sents_df, topn=topn, max_df=max_df)
    for k in keyword_clusters:
        print(k)
        keywords = keyword_clusters[k]
        print(keywords)
    return keyword_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpret changes')
    parser.add_argument('--emb_path', type=str, help='Path to the embeddings (pickel file).')
    parser.add_argument('--res_path', type=str, help='Path to the results of the clustering (csv file).')

    parser.add_argument('--method', type=str, default='K5',
                        help='Method to use to select the results for analysis: K5, K7 or AP.')
    parser.add_argument('--nb_target', type=int, default='5', help='Number of target words to select')
    parser.add_argument('--save_path', type=str, default='', help='Path to save the results.')
    args = parser.parse_args()

    if args.method == 'K7':
        method = 'kmeans_7'
    elif args.method == 'K5':
        method = 'kmeans_5'
    elif args.method == 'AP':
        method = 'aff_prop'
    target_words_path = 'aylien_target_list.txt'

    categ = "months"

    print("Selecting keywords:")
    target_words_reduc = ['strain']

    for word in target_words_reduc:
        print('##########################################', word, '################################')
        keyword_clusters = full_analysis(word, max_df, topn, method, corpus_slices, threshold_size_cluster)

    # exemple of how to use it:
    # python interpretation_aylien.py --res_path '/people/montariol/data/scalable_semantic_change/results/aylien_fox_cnn_results_english_fine_tuned_averaged.csv' --emb_path '/people/montariol/data/scalable_semantic_change/' --nb_target 5














