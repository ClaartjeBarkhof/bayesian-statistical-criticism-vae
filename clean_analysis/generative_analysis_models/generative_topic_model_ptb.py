import os
import torch
import pandas as pd
import numpy as np

# Runnning this cell for the first time requires downloading wordnet
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim.models import Phrases
from gensim.utils import any2unicode
from gensim.matutils import corpus2dense
from gensim.corpora import Dictionary
from transformers import RobertaTokenizerFast

# %config InlineBackend.figure_format='retina'
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch.distributions as td

# from analysis_steps import make_run_overview_df
# from utils import load_checkpoint_model_for_eval


# Train LDA model.
# from gensim.models import LdaModel

from .gensim_LDA import *
from random import shuffle


class GenLDATopicModelPTB:
    def __init__(self, train_samples_strings, num_topics=10, chunksize=2000, passes=20, iterations=600, eval_every=None):

        # Pre-process the train samples
        self.dictionary, self.train_corpus, _, self.train_docs, self.num_tokens = self.create_lda_corpus(
            train_samples_strings)
        _ = self.dictionary[0]  # This is only to "load" the dictionary.
        self.id2token = self.dictionary.id2token

        self.lda_model = LdaModel(
            corpus=self.train_corpus,
            id2word=self.id2token,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )

    def print_topics(self):
        topics = self.lda_model.show_topics()
        print("\nLDA model topics:\n")
        for t in topics:
            print(t[0], t[1], end='\n\n')

    @staticmethod
    def create_lda_corpus(text_samples):
        docs = []
        for text_string in text_samples:
            unicode_text_string = any2unicode(text_string, encoding='utf8', errors='strict')
            docs.append(unicode_text_string)

        # Split the documents into tokens.
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 1] for doc in docs]

        # Remove stop words.
        stop_words = set(stopwords.words('english'))
        docs = [[token for token in doc if token not in stop_words] for doc in docs]

        # Lemmatize the documents
        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

        # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
        bigram = Phrases(docs, min_count=20)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)

        # Filter out words that occur less than 20 documents, or more than 50% of the documents.
        dictionary.filter_extremes(no_below=5, no_above=0.1)

        # Bag-of-words representation of the documents.
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))

        num_tokens = len(dictionary)
        dense_corpus = corpus2dense(corpus, num_terms=num_tokens, dtype=np.int64)

        return dictionary, corpus, dense_corpus.T, docs, num_tokens

    def lda_preprocess_transform(self, text_samples):
        docs = []
        for text_string in text_samples:
            unicode_text_string = any2unicode(text_string, encoding='utf8', errors='strict')
            docs.append(unicode_text_string)

        # Split the documents into tokens.
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 1] for doc in docs]

        # Remove stop words.
        stop_words = set(stopwords.words('english'))
        docs = [[token for token in doc if token not in stop_words] for doc in docs]

        # Lemmatize the documents
        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

        # Add bigrams and trigrams to docs (all of them, dictionary will decide which ones stay)
        # TODO: check if this is correct, should it not add all bigrams then?
        bigram = Phrases(docs, min_count=1)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

        output_docs = []
        for doc in docs:
            new_doc = []
            for token in doc:
                # if token in vocab
                if token in self.dictionary.token2id:  # and token not in new_doc: (there actually may be duplicates!)
                    new_doc.append(token)
            output_docs.append(new_doc)

        # Bag-of-words representation of the documents.
        corpus = [self.dictionary.doc2bow(doc) for doc in output_docs]

        return corpus, output_docs

    def estimate_conditional_log_p_x(self, corpus, conditioning_corpus):
        """Adapted from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py"""
        # p(x*|D, theta*), with theta* being inferred from the condition
        assert len(corpus) == len(conditioning_corpus), \
            "the corpus and conditioning corpus need to be equal length, because they are interpreted as pairs"
        gamma, _ = self.lda_model.inference(conditioning_corpus)

        scores_all_docs, _ = self.lda_model.bound(corpus, gamma=gamma)

        return scores_all_docs

    def estimate_log_p_x(self, corpus, N_perm=100):
        """Adapted from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py"""

        reference_corpus = corpus.copy()

        all_scores = []
        for i in range(N_perm):
            print(f"{i:2d}/{N_perm}", end="\r")
            shuffle(reference_corpus)

            gamma, _ = self.lda_model.inference(reference_corpus)
            scores, _ = self.lda_model.bound(corpus, gamma=gamma)

            all_scores.append(np.array(scores))

        # [N_perm, N]
        all_scores = np.stack(all_scores)

        # [N]
        all_scores = torch.logsumexp(torch.FloatTensor(all_scores), dim=0).numpy() - np.log(N_perm)

        return all_scores

    def assess_surprisal_under_model(self, text_sample_dict, N_perm=50):
        unconditional_samples = text_sample_dict["unconditional_sampled_text"]
        unconditional_samples_corpus, _ = self.lda_preprocess_transform(unconditional_samples)

        conditional_samples = text_sample_dict["conditional_sampled_text"]
        conditional_samples_corpus, _ = self.lda_preprocess_transform(conditional_samples)

        conditional_condition = text_sample_dict["conditional_original_text"]
        conditional_condition_corpus, _ = self.lda_preprocess_transform(conditional_condition)

        #print("unconditional_samples_corpus[0]", unconditional_samples_corpus[0])
        #print("conditional_samples_corpus[0]", conditional_samples_corpus[0])
        #print("conditional_condition_corpus[0", conditional_condition_corpus[0])

        # p(x*|D) for unconditional samples
        unconditional_unconditional = self.estimate_log_p_x(corpus=unconditional_samples_corpus,
                                                            N_perm=N_perm)

        # p(x*|D) for conditional samples
        unconditional_conditional = self.estimate_log_p_x(corpus=conditional_samples_corpus,
                                                          N_perm=N_perm)

        # p(x*|D, theta_condition) for conditional samples
        conditional_conditional = self.estimate_conditional_log_p_x(corpus=conditional_samples_corpus,
                                                                    conditioning_corpus=conditional_condition_corpus)

        surprisal_dict = dict(
            unconditional_unconditional=unconditional_unconditional,
            unconditional_conditional=unconditional_conditional,
            conditional_conditional=conditional_conditional
        )

        return surprisal_dict