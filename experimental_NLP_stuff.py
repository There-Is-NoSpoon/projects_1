import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import StemmerI
from nltk.corpus import wordnet
from nltk import ne_chunk
from nltk import ngrams, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("maxent_ne_chunker")
nltk.download("words")


def rem_stopwords(text):
    with open("stopwords.txt", "r") as f:
        stopwords = set(f.read().splitlines())

    filtered_lines = []
    for line in text.split("\n"):
        if line == "":
            continue
        filtered_words = [
            word.lower() for word in line.split(" ") if word.lower() not in stopwords
        ]
        filtered_line = " ".join(filtered_words)
        filtered_lines.append(filtered_line)
    return filtered_lines


def read_folder(folder_name):
    file_dict = {}
    for filename in os.listdir(folder_name):
        file_path = os.path.join(folder_name, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                text = file.read()
                file_dict[filename] = text
    return file_dict


corpus = {
    "c1": read_folder(
        r"C:\Users\jlee4\OneDrive\Desktop\PredictiveAnalytics\hw3\data\C1"
    ),
    "c4": read_folder(
        r"C:\Users\jlee4\OneDrive\Desktop\PredictiveAnalytics\hw3\data\C4"
    ),
    "c7": read_folder(
        r"C:\Users\jlee4\OneDrive\Desktop\PredictiveAnalytics\hw3\data\C7"
    ),
}

post_stop_corpus = {}

for folder, articles in corpus.items():
    post_stop_corpus[folder] = {}
    for article, text in articles.items():
        post_stop_corpus[folder][article] = rem_stopwords(text)


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]

    ps = PorterStemmer()
    # tokens_postStem = [ps.stem(token) for token in tokens]

    lz = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens:
        pos = nltk.pos_tag([token])[0][1][0].upper()
        pos = pos if pos in ["n", "v", "r", "j"] else wordnet.NOUN
        lemmatized_token = lz.lemmatize(token, pos)
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens


def apply_ner(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    chunked_tokens = ne_chunk(tagged_tokens)

    merged_tokens = []
    for subtree in chunked_tokens:
        if hasattr(subtree, "label") and isinstance(subtree, nltk.Tree):
            merged_token = "_".join([token.lower() for token, _ in subtree.leaves()])
            merged_tokens.append(merged_token)
        else:
            if isinstance(subtree, tuple):
                merged_tokens.append(subtree[0])
            else:
                merged_tokens.extend([token.lower() for token, _ in subtree.leaves()])

    return merged_tokens


def merge_bigrams(tokens):
    res = [
        (x, i.split()[j + 1])
        for i in tokens
        for j, x in enumerate(i.split())
        if j < len(i.split()) - 1
    ]

    bigram_frequencies = FreqDist(ngrams(tokens, 2))
    merged_tokens = []

    i = 0
    for i in range(len(tokens) - 1):
        bigram = tuple(tokens[i : i + 2])
        if bigram_frequencies[bigram] >= 1:
            merged_tokens.append(" ".join(bigram))
    return merged_tokens


merged_tokens = {}
for folder, articles in post_stop_corpus.items():
    merged_tokens[folder] = {}
    for article, text in articles.items():
        merged_tokens[folder][article] = merge_bigrams(
            apply_ner(tokenize(" ".join(text)))
        )

all_tokens = []

for folder, articles in merged_tokens.items():
    for article, text in articles.items():
        all_tokens += merged_tokens[folder][article]

index = []
for folder, articles in merged_tokens.items():
    index += [(folder, article) for article in articles.keys()]

document_term_matrix = pd.DataFrame(
    columns=all_tokens, index=pd.MultiIndex.from_tuples(index)
).fillna(0)
document_term_matrix

for folder, articles in merged_tokens.items():
    for article, text in articles.items():
        for token in text:
            document_term_matrix.loc[(folder, article), token] += 1

tf_matrix = document_term_matrix.div(document_term_matrix.sum(axis=1), axis=0)
idf = np.log10(len(document_term_matrix) / document_term_matrix.astype(bool).sum())

tf_idf_matrix = (
    tf_matrix.mul(idf).replace(0.0, np.nan).dropna(how="all", axis=1).fillna(0)
)
tf_idf_matrix.sum().sort_values().head(20)
#print(document_term_matrix)
#print(tf_idf_matrix)

#print(tf_idf_matrix.index)

def calculate_cosine_similarity(tfidf_matrix, doc_name):
    doc_index = tfidf_matrix.index.get_loc(doc_name)
    doc_tfidf = tfidf_matrix.iloc[doc_index]

    # Calculate cosine similarity between the given document and all other documents
    similarities = cosine_similarity([doc_tfidf], tfidf_matrix)

    # Convert similarities array into a DataFrame
    similarities_df = pd.DataFrame(similarities, columns=tfidf_matrix.index)

    # Sort the DataFrame by similarity in descending order
    sorted_similarities_df = similarities_df.transpose().sort_values(by=0, ascending=False)

    return sorted_similarities_df

result = calculate_cosine_similarity(tf_idf_matrix, ('c1', 'article01.txt'))

print(result)

#document input in format as ('c1', 'article01.txt')
def kNN(tf_idf_matrix, document, k):
    doc_tfidf = tf_idf_matrix.loc[[document]]

    # Calculate cosine similarity between the given document and all other documents
    similarities = cosine_similarity(doc_tfidf, tf_idf_matrix)

    # Convert similarities array into a DataFrame
    similarities_df = pd.DataFrame(similarities, columns=tf_idf_matrix.index)

    # Sort the DataFrame by similarity in descending order
    sorted_similarities_df = similarities_df.transpose().sort_values(by=0, ascending=False)

    # Get the top K highest cosine similarity values and their corresponding documents
    k_nearest_documents = sorted_similarities_df.iloc[1:k+1]

    return k_nearest_documents

print(kNN(tf_idf_matrix, ('c4', 'article03.txt'), 5))