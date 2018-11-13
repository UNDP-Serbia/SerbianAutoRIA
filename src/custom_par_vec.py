from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class CustomParVec:
    """
        Custom Paragraph Vector. Each paragraph (or sentence) is the sum of each of its word's Word2Vec vector
        representation scaled by that word's tf(-idf).
    """

    def __init__(self, word_sentence_list, workers=10, dimensions=300, min_word_count=30, context=30, sg=0,
                 iterations=5, downsampling=0, tfidf=True, target_docs=None):
        """
        Args:
            word_sentence_list (list[list[str]]) : List of lists of words that make up a paragraph(or sentence).
            workers (int)                        : Number of threads to run in parallel (Default = 10).
            dimensions (int)                     : Vector dimensionality (Default = 300).
            min_word_count (int)                 : Minimum word count (Default = 30).
            context (int)                        : Context window size (Default = 30).
            sg(int)                              : Training algorithm 0-CBOW, 1-Skipgram (Default = CBOW).
            iterations(int)                      : Number of iterations over the corpus (Default = 5).
            downsampling (int)                   : Downsample setting for frequent words (Default = 0).
            tfidf (bool)                         : Specify whether or not to use idf in scaling (Default = True).
            target_docs (list[str])              : List of documents for each SDG target (Default = None).
        """
        
        self._dimensions = dimensions
        self._word2vec_model = Word2Vec(word_sentence_list, workers=workers, size=self._dimensions, sg=sg,
                                        iter=iterations, min_count=min_word_count, window=context, sample=downsampling)
        self._word2vec_model.init_sims(replace=True)  # used for memory efficiency
        
        self._sentences = [' '.join(words) for words in word_sentence_list]  # Retain sentences for tfidf calculations

        self._corpus_tf_idf_obj = None
        self._corpus_tf_idf = None
        self._corpus_word_index = None
        self._target_tf_idf_obj = None
        self._target_tf_idf = None
        self._target_word_index = None

        self.set_corpus_tfidf(tfidf)
        self.set_target_docs(tfidf, target_docs)

    def set_corpus_tfidf(self, tfidf):
        """
            Enable/disable TFIDF scaling, using corpus-based TFIDF

        Args:
            tfidf (bool): If true, CustomParVec will use corpus tfidf in word2vec scaling.
             If false, corpus nbow is used.
        """
        self._corpus_tf_idf_obj = TfidfVectorizer(use_idf=tfidf)  # Create TfidfVectorizer object
        # Transform and fit tf-idf to all sentences (could be paragraphs)
        self._corpus_tf_idf = self._corpus_tf_idf_obj.fit_transform(self._sentences)
        # Keep track of words by index for lookups
        self._corpus_word_index = self._corpus_tf_idf_obj.get_feature_names()
        self._target_tf_idf_obj = None

    def set_target_docs(self, tfidf, target_docs):
        """
            Enable/disable TFIDF scaling, using target-based TFIDF

        Args:
            tfidf (bool): If true, CustomParVec will use tfidf in word2vec scaling, if false nbow will be used.
             The source for tfidf is determined via the target_docs argument.
            target_docs (list[str]): The list of target documents to be used in TFIDF calculations
        """
        if target_docs:
            self._target_tf_idf_obj = TfidfVectorizer(use_idf=tfidf)  # Create TfidfVectorizer object
            # Transform and fit tf-idf to all sentences(could be paragraphs)
            self._target_tf_idf = self._target_tf_idf_obj.fit_transform(target_docs)
            # Keep track of words by index for lookups
            self._target_word_index = self._target_tf_idf_obj.get_feature_names()
        else:
            self._target_tf_idf_obj = None

    def get_most_similar(self, sentences, vectors, sentence, top_n=10, threshold=0.5):
        """
            Given a new sentence, find the closest top_n elements
 
        Args:
            sentences (list[str])         : List of sentences to be compared to
            vectors (list[numpy.ndarray]) : Vector embedding of sentences
            sentence(str)                 : Text we want to find most similar to.
            top_n (int)                   : Total number of most similar tuples we want returned (Default = 10).
            threshold (float)             : Minimum Cosine Distance to be returned (Default = 0.5)

        Returns: 
            list[(float, string)]: A list of (cosine similarity, sentence) tuples of size top_n closest to the 
                                     input sentence.
        """
        inferred_vector = self.infer_vector(sentence)
            
        cos_similarities = np.ravel(cosine_similarity(inferred_vector.reshape(1, -1), vectors))
        # sorts the indices of the top_n most similar sentences
        most_similar = np.argpartition(-cos_similarities, top_n)[:top_n]
        return [(cos_similarities[sentence_index], sentences[sentence_index])
                for sentence_index in most_similar if cos_similarities[sentence_index] >= threshold]
                           
    def infer_vector(self, line):
        """
            Generate a numerical vector representation for the given line of text.
            If a set of target documents has been provided, use them for tfidf calculations, if tfidf is enabled.
            Otherwise, use corpus tfidf, if tfidf is enabled.

        Args:
            line (str): text whose vector representation is to be inferred
        Returns:
            inferred vector (numpy.ndarray) : vector representation of the given text
        """
        if self._target_tf_idf_obj:
            return self._infer_vector_target_based(line)
        return self._infer_vector_corpus_based(line)

    def _infer_vector_corpus_based(self, line):
        """
            Given a new line, infer a custom vector representation using the corpus tfidf.
 
        Args: 
            line (str): text whose vector representation is to be inferred
        Returns: 
            inferred vector (numpy.ndarray) : vector representation of the given text
        """
        line_tf_idf = self._corpus_tf_idf_obj.transform([line])  # infer the tf-idf values for the words in the line
        rows, cols = line_tf_idf.nonzero()
        
        new_vec = np.zeros(self._dimensions)
        for col in cols:
            try:    
                new_vec += (self._word2vec_model[(self._corpus_word_index[col])] * line_tf_idf[0, col])
            except:
                continue
        return np.asarray(new_vec)
    
    def _infer_vector_target_based(self, line):
        """
            Given a new line, infer a custom vector representation using the target documents tfidf.
 
        Args: 
            line (str): text whose vector representation is to be inferred
        Returns: 
            inferred vector (numpy.ndarray) : vector representation of the given text
        """

        """ for each word that exists in a national policy document but not in a RIA, we look for words that are close
         to it in the Word2Vec vectorspace, and take the tf-idf score of the most similar word that is available.
         If no such word is available, the scaling factor is set to 0."""
        replacement_words = []
        for word in line.split():
            if word not in self._target_tf_idf_obj.vocabulary_:
                try:
                    similar_words = self._word2vec_model.similar_by_word(word, topn=10, restrict_vocab=None)
                    for sim in similar_words:
                        if sim[0] in self._target_tf_idf_obj.vocabulary_:
                            replacement_words.append((word, sim[0]))
                            break
                except:
                    continue
                    
        for old, new in replacement_words:
            line = line.replace(old, new)
            
        line_tf_idf = self._target_tf_idf_obj.transform([line])  # infer the tf-idf values for the words in the line
        rows, cols = line_tf_idf.nonzero()
        
        new_vec = np.zeros(self._dimensions)
        # Apply the same sentence to vector conversion as above. 
        for col in cols:
            try:    
                new_vec += (self._word2vec_model[(self._target_word_index[col])] * line_tf_idf[0, col])
            except:
                continue
        return np.asarray(new_vec)
