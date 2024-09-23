"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
import os
from collections import defaultdict, Counter
import pickle
import math
import operator

from tqdm import tqdm
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

class Indexer:
    dbfile = "./ir.idx"  # file to save/load the index

    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = {}
        self.postings_lists = defaultdict(list)
        self.docs = []
        self.corpus_stats = {'avgdl': 0, 'doc_lengths': []}
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')

        if os.path.exists(self.dbfile):
            print("Old index file detected, deleting and recreating index...")
            os.remove(self.dbfile)  # Delete old index file to force recreation

        print("Loading dataset...")
        ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
        self.raw_ds = ds['article']
        self.clean_text_and_index(self.raw_ds)  # Preprocess and create index


    def clean_text_and_index(self, lst_text):
        """ Tokenize, lemmatize and build postings list """
        print("Indexing documents...")
        for doc_id, doc in tqdm(enumerate(lst_text)):
            tokens = self.clean_text(doc)
            self.docs.append(tokens)  # Store processed tokens for each document
            self.corpus_stats['doc_lengths'].append(len(tokens))  # Document length

            # Build the inverted index
            for token in set(tokens):  # Use set to get unique tokens
                term_frequency = tokens.count(token)
                if token not in self.tok2idx:
                    idx = len(self.tok2idx)
                    self.tok2idx[token] = idx
                    self.idx2tok[idx] = token
                self.postings_lists[token].append((doc_id, term_frequency))

        # Calculate average document length
        self.corpus_stats['avgdl'] = sum(self.corpus_stats['doc_lengths']) / len(lst_text)

        # Save the index for future use
        self.save_index()

    def clean_text(self, text):
        """ Clean, tokenize and lemmatize text """
        tokens = self.tokenizer.tokenize(text.lower())  # Tokenize by word
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stopwords]  # Lemmatize and remove stopwords
        return tokens

    def save_index(self):
        """ Save index and metadata using pickle """
        with open(self.dbfile, 'wb') as f:
            pickle.dump((self.tok2idx, self.idx2tok, self.postings_lists, self.corpus_stats, self.raw_ds), f)

    def load_index(self):
        """ Load the index and metadata """
        with open(self.dbfile, 'rb') as f:
            self.tok2idx, self.idx2tok, self.postings_lists, self.corpus_stats, self.raw_ds = pickle.load(f)


class SearchAgent:
    k1 = 1.5  # BM25 parameter k1 for tf saturation
    b = 0.75  # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        self.i = indexer

    def query(self, q_str):
        """ Process query, compute BM25 scores, and display results """
        query_tokens = self.i.clean_text(q_str)  # Preprocess the query
        doc_scores = defaultdict(float)

        # Calculate BM25 score for each document
        for token in query_tokens:
            if token in self.i.postings_lists:
                idf = self.compute_idf(token)
                for doc_id, freq in self.i.postings_lists[token]:
                    doc_len = self.i.corpus_stats['doc_lengths'][doc_id]
                    avgdl = self.i.corpus_stats['avgdl']
                    score = idf * ((freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)))
                    doc_scores[doc_id] += score

        # Sort documents by their BM25 score in descending order
        sorted_docs = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)
        self.display_results(sorted_docs)

    def compute_idf(self, token):
        """ Compute inverse document frequency (IDF) """
        N = len(self.i.docs)  # Total number of documents
        df = len(self.i.postings_lists[token])  # Document frequency of the token
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def display_results(self, results):
        """ Display top results """
        for docid, score in results[:5]:  # Display top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid])


if __name__ == "__main__":
    # Instantiate the indexer and search agent
    i = Indexer()  # Create an indexer object
    q = SearchAgent(i)  # Pass the indexer to the search agent

    # Simulate an interactive querying session using a loop
    while True:
        # Ask for user input (query)
        query = input("\nEnter your search query (or type 'exit' to quit): ")
        
        # If the user types 'exit', break the loop and stop the program
        if query.lower() == 'exit':
            print("Exiting the search engine...")
            break

        # Perform the search using the provided query
        q.query(query)
