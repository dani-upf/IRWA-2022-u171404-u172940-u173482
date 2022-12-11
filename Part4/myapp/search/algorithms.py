from array import array
from collections import defaultdict
import math
import numpy as np
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import string

# 1. create create_tfidf_index
import random
from myapp.search.objects import ResultItem, Document

def create_inverted_index(corpus):
    inverted_index = defaultdict(list)

    for key, doc in corpus.items():
        terms = doc.description

        current_page_index = {}

        for position, term in enumerate(terms):  # Loop over all terms
            try:
                # if the term is already in the index for the current page (current_page_index)
                # append the position to the corresponding list (elemento 1 del arreglo, el 0 es la id del documento)
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term] = [doc.id,
                                            array('I', [position])]  # 'I' indicates unsigned int (int in Python)

        # merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            inverted_index[term_page].append(posting_page)

    return inverted_index


def _clean_tweet(line, emojis):
    """
    Preprocess the article text (title + body) removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    line = str(line)  # by default everything assumed as string

    line = line.lower()  # transform into lower case

    line = re.sub(r'\\n', ' ', line)  # remove new lines

    line = line.translate(str.maketrans('', '', string.punctuation))  # remove punctuation marks

    line = re.sub(r'http\S+', '', line)  # remove links

    if emojis:  # remove emojis
        ### https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
        remove_emojis = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   u"\u2026"  # ...
                                   u"\u2019"  # '
                                   u"\u2066"
                                   u"\u2069"
                                   u"\u231b"
                                   "]+", flags=re.UNICODE)
        line = remove_emojis.sub(r'', str(line))

    line = line.split()  # Tokenize the text to get a list of terms

    stop_words = set(stopwords.words("english"))  # remove the stopwords
    line = [x for x in line if x not in stop_words]

    stemmer = PorterStemmer()  # stem terms
    line = [stemmer.stem(word) for word in line]

    return line

def check(list1, val):
    return(all(x > val for x in list1))

def search_in_corpus(corpus:dict, search_id, query):
    """
        Helper method, just to demo the app
        :return: a list of demo docs sorted by ranking
        """
    # 1. create create_tfidf_index
    #Añadido
    for key, doc in corpus.items():
        corpus[key].description = _clean_tweet(doc.description, True)

    inverted_index = create_inverted_index(corpus)
    query_terms = _clean_tweet(query, False)
    tf_idf = {}

    for key, doc in corpus.items():
        tf_idf[doc.id] = [0.0] * len(inverted_index)  # initialize document vectors

    for index, term in enumerate(inverted_index):
        df = len(inverted_index[term])  # document frequency of the term

        for doc in inverted_index[term]:
            f = len(doc[1])  # term frequency of the term in the document doc

            w = (1 + math.log(f, 2)) * math.log(len(corpus) / df, 2)
            tf_idf[doc[0]][index] = w

    tf_idf_q = [0.0] * len(inverted_index)  # initialize document vectors

    for index, term in enumerate(inverted_index):
        df = len(inverted_index[term])  # document frequency of the term
        f =query_terms.count(term)  # term frequency of the term in the query q
        if(f==0):#if frequency is 0, weight is 0
            tf_idf_q[index] = 0
        else:
            w = (1 + math.log(f, 2)) * math.log(len(corpus) / df, 2)
            tf_idf_q[index] = w

    cosine = {}
    for d in tf_idf.keys():
        doc = tf_idf[d]

        #cosine[d] = np.dot(tf_idf_q, doc) / (norm(tf_idf_q) * norm(doc))  # compute cosine similarity
        #more efficeint
        cosine[d]=0
        vec_size=len(doc)
        for i in range(vec_size):
            if(tf_idf_q[i]!=0):
                cosine[d]=cosine[d]+(tf_idf_q[i]*doc[i])
        if(cosine[d]!=0):
            cosine[d]=cosine[d]/ (norm(tf_idf_q) * norm(doc))

    cosine = dict(sorted(cosine.items(), key=lambda x: x[1], reverse=True))  # sort cosine similarity in descending order

    top_k=10
    k_ranking=[]
    cosine_ids=list(cosine.keys())
    #print(cosine_ids)
    for ids in cosine_ids[:top_k]:
        #print(ids)
        k_ranking.append(ids)

    #THE K RANKING WIL BE DISPLAAYED IN INVERSE ORDER, SO WE REVERSE IT
    k_ranking.reverse()


    res = []
    size = len(corpus)
    ll = list(corpus.values())


    for index in range(top_k):
        id=k_ranking[index]#index del top document
        item: Document = corpus.get(id)#op document
        res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                              "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id),index))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


