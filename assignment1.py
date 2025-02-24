import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import pprint


class CustomTFIDFvectoriser(CountVectorizer):

    def fit_transform(self, raw_documents, y=None):
        tfidf_vectors = super().fit_transform(raw_documents)
        doc_rows, feature_columns = tfidf_vectors.shape
        arr = tfidf_vectors.toarray()
        transposed_tfidf_vectors = tfidf_vectors.T
        transposed_arr = transposed_tfidf_vectors.toarray()
        '''doc freq is for document frequency, number of documents containing the term, len is 1400'''
        docfreq = []
        for m in range(feature_columns):
            doc_freq_count = 0
            for n in range(doc_rows):
                if transposed_arr[m, n] != 0:
                    doc_freq_count = doc_freq_count + 1
            docfreq.append(doc_freq_count)
        '''wordcount_docs has the list of total number of words in in each doc'''
        wordcount_docs = np.sum(arr, axis=1).tolist()
        tf_arr = np.ones((doc_rows, feature_columns))
        for x in range(doc_rows):
            for y in range(feature_columns):
                if arr[x, y] == 0 and wordcount_docs[x] == 0:
                    tf_arr[x, y] = 1
                elif arr[x, y] == 0 and wordcount_docs[x] != 0:
                    tf_arr[x, y] = (2 / (1 + np.log(wordcount_docs[x])))
                elif (wordcount_docs[x] == 0) and (arr[x, y] != 0):
                    tf_arr[x, y] = ((1 + (np.log(arr[x, y]))) / 2)
                else:
                    tf_arr[x, y] = (1 + np.log(arr[x, y])) / (1 + np.log(wordcount_docs[x]))

        inversedocfreq = [1] * doc_rows
        for a in range(doc_rows):
            if docfreq[a] != 0:
                inversedocfreq[a] = 1 / docfreq[a]
        tfidf_arr = np.ones((doc_rows, feature_columns))
        for b in range(doc_rows):
            for c in range(feature_columns):
                tfidf_arr[b, c] = inversedocfreq[b] * tf_arr[b, c]

        return tfidf_arr


with open('query.text', 'r') as file:
    queryPath = file.read()

with open('cran.all', 'r') as file:
    cranFile = file.read()

with open('qrels.text', 'r') as file:
    qrelsPath = file.readlines()
'''Convert a file from txt to dictionary'''


def convert_file_dict(data):
    query_dict = {}
    query_id = data.split('.I')
    for k in query_id:
        if len(k) > 1:
            only_doc_matter = k.split('.W')
            unique_id = only_doc_matter[0].strip().split()[0]
            description = only_doc_matter[1]
            description = ' '.join(word for word in description.split() if word.lower() not in ENGLISH_STOP_WORDS)
            lowered_description = description.lower()
            query_dict[int(unique_id)] = lowered_description

    return query_dict


'''Convert all the values in dict to list that only contains all the values'''


def only_matter(id_matter_dict):
    matter_list = [id_matter_dict[idKey] for idKey in id_matter_dict]
    return matter_list


def relevancescores(weighs, relevant, query_data_len):
    pr = []
    re = []
    f1val = []

    for i in range(query_data_len):
        retrivedelement = weighs[i]
        relevantelement = relevant.get(i+1, set())

        tp = len(set(retrivedelement) & set(relevantelement))

        p = tp / len(retrivedelement) if len(retrivedelement) > 0 else 0
        r = tp / len(relevantelement) if len(relevantelement) > 0 else 0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

        pr.append(p)
        re.append(r)
        f1val.append(f)

    return pr, re, f1val

def qrelationFile(matter):
    query_relevant_docs = {}
    for line in matter:
        arr = line.split()
        query_id = int(arr[0])
        document_id = int(arr[1]) - 1
        if query_id not in list(query_relevant_docs.keys()):
            query_relevant_docs[query_id] = []
        query_relevant_docs[query_id].append(document_id)
    return query_relevant_docs


def plotting_graphs(name, yValues, yName):
    plt.figure()
    plt.title(name)
    plt.xlabel("Query Index")
    plt.ylabel(yName)
    plt.grid(True)
    plt.plot(range(1, len(yValues) + 1), yValues)
    plt.savefig(name)


'''All the values in dictionary is transferred to a list'''
id_document = convert_file_dict(cranFile)
cran_matter_list = only_matter(id_document)
vectorizer = CountVectorizer(binary=True, lowercase=True, stop_words=list(ENGLISH_STOP_WORDS))
vectors = vectorizer.fit_transform(cran_matter_list)
binaryVectors = vectors.toarray()

''' Query Processing'''
query_id_document = convert_file_dict(queryPath)
query_ids = list(query_id_document.keys())
query_matter_list = only_matter(query_id_document)
query_vectors = vectorizer.transform(query_matter_list)
query_binaryVectors = query_vectors.toarray()

'''cosine similarity & Pairwise'''
cosine = cosine_similarity(query_vectors, vectors)
euclidean_distance = pairwise_distances(query_vectors, vectors, metric='euclidean')
euclidean_final_rank = np.argsort(euclidean_distance, axis=1)[:, :10]
final_rank = np.argsort(cosine, axis=1)[:, ::-1][:, :10]

comparator = qrelationFile(qrelsPath)

bin_cos_pr, bin_cos_re, bin_cos_f1 = relevancescores(final_rank, comparator, len(query_ids))
bin_euc_pr, bin_euc_re, bin_euc_f1 = relevancescores(euclidean_final_rank, comparator, len(query_ids))

tfidf = CustomTFIDFvectoriser()
tfidf_vec = tfidf.fit_transform(cran_matter_list)
tfidf_query_vec = tfidf.transform(query_matter_list)

'''Cosine similarity'''
tfidf_c = cosine_similarity(tfidf_query_vec, tfidf_vec)
tfidf_final_rank = np.argsort(tfidf_c, axis=1)[:, ::-1][:, :10]

'''Euclidean'''
tfidf_euclidean_distance = pairwise_distances(tfidf_query_vec, tfidf_vec, metric='euclidean')
tfidf_euclidean_final_rank = np.argsort(tfidf_euclidean_distance, axis=1)[:, :10]

tfidf_cos_pr, tfidf_cos_re, tfidf_cos_f1 = relevancescores(tfidf_final_rank, comparator, len(query_ids))
tfidf_euc_pr, tfidf_euc_re, tfidf_euc_f1 = relevancescores(tfidf_euclidean_final_rank, comparator, len(query_ids))

plotting_graphs('Binary Cosine Precision values for Top 10 docs', bin_cos_pr, "Precision")
plotting_graphs('Binary Cosine Recall values for Top 10 docs', bin_cos_re, "Recall")
plotting_graphs('Binary Cosine F1 values for Top 10 docs', bin_cos_f1, "F1-score")

plotting_graphs('Binary Euclidean Precision values for Top 10 docs', bin_euc_pr, "Precision")
plotting_graphs('Binary Euclidean Recall values for Top 10 docs', bin_euc_re, "Recall")
plotting_graphs('Binary Euclidean F1 values for Top 10 docs', bin_euc_f1, "F1-score")

plotting_graphs('TFIDF Cosine Precision values for Top 10 docs', tfidf_cos_pr, "Precision")
plotting_graphs('TFIDF Cosine Recall values for Top 10 docs', tfidf_cos_re, "Recall")
plotting_graphs('TFIDF Cosine F1 values for Top 10 docs', tfidf_cos_f1, "F1-score")

plotting_graphs('TFIDF Euclidean Precision values for Top 10 docs', tfidf_euc_pr, "Precision")
plotting_graphs('TFIDF Euclidean Recall values for Top 10 docs', tfidf_euc_re, "Recall")
plotting_graphs('TFIDF Euclidean F1 values for Top 10 docs', tfidf_euc_f1, "F1-score")

res = {
    'Binary': {'f': {'cos': (np.mean(bin_cos_f1), max(bin_cos_f1)),
                     'euc': (np.mean(bin_euc_f1), max(bin_euc_f1))},
               'p': {'cos': (np.mean(bin_cos_pr), max(bin_cos_pr)),
                     'euc': (np.mean(bin_euc_pr), max(bin_euc_pr))},
               'r': {'cos': (np.mean(bin_cos_re), max(bin_cos_re)),
                     'euc': (np.mean(bin_euc_re), max(bin_euc_re))}
               },
    'TFIDF': {'f': {'cos': (np.mean(tfidf_cos_f1), max(tfidf_cos_f1)),
                    'euc': (np.mean(tfidf_euc_f1), max(tfidf_euc_f1))},
              'p': {'cos': (np.mean(tfidf_cos_pr), max(tfidf_cos_pr)),
                    'euc': (np.mean(tfidf_euc_pr), max(tfidf_euc_pr))},
              'r': {'cos': (np.mean(tfidf_cos_re), max(tfidf_cos_re)),
                    'euc': (np.mean(tfidf_euc_re), max(tfidf_euc_re))}
              }
}

pprint.pprint(res)
