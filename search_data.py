import os
import re
import gensim
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity

# ------------------------------------------------------------------------

# Data Preprocessing: This function cleans and processes the input data
def processed_corpus(data):
    # Read stopwords from file
    with open("gist_stopwords.txt", "r") as file:
        stopwords = file.read().split(",")
    
    # Extract 'Name' and 'Comment' columns from the input data
    df = data[['Name', 'Comment']].copy()
    names_list = df['Name'].tolist()
    comments_list = df['Comment'].tolist()

    # Clean comments: Remove non-alphanumeric characters and convert to lowercase
    comments_list = [re.sub(r'\W+', ' ', comment).strip().lower() for comment in comments_list]

    # Function to split camelCase words into individual words
    def camel_case_split(input_str):
        return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', input_str)

    # Process names to handle camelCase and underscore-separated words
    for idx, name in enumerate(names_list):
        camel_case_words = camel_case_split(name)
        if not camel_case_words:  # If no camelCase is found, split by underscores
            names_list[idx] = name.split("_")
        else:
            names_list[idx] = camel_case_words

    # Combine names and comments into a single list of lowercased strings
    name_join_comment = [f"{' '.join(name)} : {comment}".lower() for name, comment in zip(names_list, comments_list)]

    # Remove stopwords from the combined name-comment strings
    corpus = [[word for word in entry.split()[:13] if word not in stopwords] for entry in name_join_comment]

    # Replace empty strings with 'INVALID STRING' to preserve matching with ground truth
    processed_corpus = [[word if word else "INVALID STRING" for word in document] for document in corpus]

    # Remove any colon characters from the corpus
    processed_corpus = [[item for item in inner_list if item != ":"] for inner_list in processed_corpus]

    # Create a frequency dictionary of words
    word_freq = {}
    for document in processed_corpus:
        for word in document:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Filter words that appear more than once (removing single occurrence words)
    filtered_word_freq = {word: count for word, count in word_freq.items() if count > 1}

    # Create the frequency vector from the filtered word frequency dictionary
    frequency_vector = list(filtered_word_freq.values())

    # Update the processed corpus to include only frequent words
    processed_corpus = [[word for word in document if word in filtered_word_freq] for document in processed_corpus]

    return processed_corpus

# ------------------------------------------------------------------------

# Frequency-Based Search: This function calculates similarity based on word frequency
def search_freq(data, query, processed_corpus):
    # Create a dictionary and a BOW representation of the corpus
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    query_bow = dictionary.doc2bow(query.split())

    # Calculate similarity between the query and the corpus
    corpus_index = MatrixSimilarity(corpus_bow)
    sims = corpus_index[query_bow]

    # Sort the similarities in descending order and return the top 5 results
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return sims[:5]

# ------------------------------------------------------------------------

# TF-IDF Based Search: This function calculates similarity using TF-IDF
def search_tfidf(data, query, processed_corpus):
    # Create a dictionary and a BOW representation of the corpus
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = TfidfModel(corpus_bow)
    
    # Create a SparseMatrixSimilarity index for the corpus
    index = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))
    query_bow = dictionary.doc2bow(query.split())

    # Calculate similarity between the query and the corpus
    sims = index[tfidf[query_bow]]

    # Sort the similarities in descending order and return the top 5 results
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return sims[:5]

# ------------------------------------------------------------------------

# Latent Semantic Indexing (LSI) Based Search: This function calculates similarity using LSI
def search_lsi(data, query, processed_corpus, return_corpus=False):
    # Create a dictionary and a BOW representation of the corpus
    dictionary = Dictionary(processed_corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in processed_corpus]
    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    # Create the LSI model and transform the corpus
    lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]  # Corpus is now transformed into LSI space
    
    # Return the corpus_lsi if requested (useful for TSNE visualization)
    if return_corpus:
        return corpus_lsi
    
    # Transform the query into LSI space and calculate similarity
    query_bow = dictionary.doc2bow(query.split())
    query_lsi = lsi[tfidf[query_bow]]
    index = MatrixSimilarity(corpus_lsi)  # Create an index for the LSI vectors
    sims = index[query_lsi]

    # Sort the similarities in descending order and return the top 5 results
    sims = abs(sims)
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    return sims[:5]

# ------------------------------------------------------------------------

# Doc2Vec Model: Read and preprocess the corpus for the doc2vec model
def read_corpus(fname, tokens_only=False):
    for i, line in enumerate(fname):
        tokens = gensim.utils.simple_preprocess(str.join(" ", line))
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

# Doc2Vec Based Search: This function calculates similarity using Doc2Vec
def search_doc2vec(data, query, processed_corpus, return_model=False):
    # Create the training corpus for the Doc2Vec model
    train_corpus = list(read_corpus(processed_corpus))

    # Load the pre-trained model if it exists, or train a new model
    if os.path.exists("doc2vec.model"):
        model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
    else:
        model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=500)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("doc2vec.model")

    # Return the trained model if requested (useful for TSNE visualization)
    if return_model:
        return model

    # Infer the vector for the query and calculate similarity
    inferred_vector = model.infer_vector(query.split())
    sims = model.dv.most_similar([inferred_vector], topn=5)

    # Return the top 5 most similar results
    return sims[:5]

