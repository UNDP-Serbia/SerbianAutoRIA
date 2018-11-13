import os
from load_data import DataLoader
from custom_par_vec import CustomParVec
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xlsxwriter


class RIA:
    """
        Contains the methods used in performing the automated RIA algorithm.
    """

    NUMBER_OF_TARGETS = 47  # Total number of SDG targets considered within RIA experiments

    def __init__(self, sdg_target_definitions_path='../data/SDG target definitions.xlsx',
                 sdg_target_document_matches_path='../data/SDG target - document text matches.xlsx',
                 training_test_split_path='../data/Training and test set division.xlsx',
                 documents_path='E:/documents3/',
                 results_path='../results/',
                 stemming=True,
                 embedding_dimensionality=300,
                 threads=10
                 ):
        print('--------------------------------------------------')
        print('Starting RIA experiments: embedding size ' + str(embedding_dimensionality) + ', stemming: '
              + ('yes' if stemming else 'no'))
        print('--------------------------------------------------')
        self._stemming = stemming
        self._embedding_dims = embedding_dimensionality
        self._data_loader = DataLoader(sdg_target_definitions_path, sdg_target_document_matches_path,
                                       training_test_split_path, stemming)
        self._target_docs = None
        corpus = list(self._data_loader.read_corpus(path=documents_path))
        print('Creating word embeddings')
        self._w2v = CustomParVec(corpus, workers=threads, dimensions=embedding_dimensionality, min_word_count=30,
                                 context=30, sg=0, iterations=5, downsampling=0, tfidf=False, target_docs=None)
        self._results_dir = results_path + str(embedding_dimensionality) + '/'
        if stemming:
            self._results_dir += 'stem/'
        else:
            self._results_dir += 'no_stem/'
        os.makedirs(self._results_dir, exist_ok=True)
        self._score_dict = None
        self._matches_by_sent = None
        self._avg_matches_by_sent = None
        self._avg_sdg_matches_by_sent = None

    def get_embedding_dims(self):
        """
            Get the size of the custom_par_vec/word2vec embeddings used in this RIA run.

        Returns:
            dim (int) : Embedding dimensionality
        """
        return self._embedding_dims

    def uses_stemming(self):
        """
            Gets the status of stemming usage in this RIA run.

        Returns:
            stemming (bool) : Whether stemming is used in this RIA run or not.
        """
        return self._stemming

    def run_ria(self, label, use_target_indicators, use_training_set_matches, tfidf, target_tfidf):
        """
            Perform one run of the AutoRIA algorithm, using the provided settings, and return the results.

        Args:
            label (str): User label assigned to this run of the RIA algorithm
            use_target_indicators (bool): Whether to include SDG target indicators in target documents
            use_training_set_matches (bool): Whether to include target matches from the training set in target documents
            tfidf (bool): Whether to use TFIDF scaling in word similarity calculations
            target_tfidf (bool): Whether to use TFIDF values from target documents (if false, the entire document corpus
                is used to derive TFIDF values)
        Returns:
            matches_by_sent, avg_matches_by_sent (tuple(dict[str], list[float])) : the first element of the tuple is
                a dictionary of per-target system performance levels (target : list[float]), and the second is a list
                containing the average, global performance levels. Each element of a performance list represents
                system performance for a particular number of sentences produced as candidates for an SDG target.
        """
        self._target_docs = None
        self._score_dict = None
        self._matches_by_sent = None
        self._avg_matches_by_sent = None
        self._avg_sdg_matches_by_sent = None
        print('--------------------------------------------------')
        print('Running RIA: ' + ('NBOW' if not tfidf else 'TFIDF' if not target_tfidf else 'Target TFIDF'))
        print('Using target indicators: ' + ('yes' if use_target_indicators else 'no'))
        print('Using training set matches: ' + ('yes' if use_training_set_matches else 'no'))
        self._target_docs = self._data_loader.create_target_documents(use_target_indicators, use_training_set_matches)
        self._w2v.set_corpus_tfidf(tfidf)
        if target_tfidf:
            target_tfidf_doc = self._target_docs.values()
            self._w2v.set_target_docs(tfidf, list(target_tfidf_doc))

        print('Performing RIA matching')
        targs, targ_vecs, sents = self._get_target_embeddings()
        self._score_dict = self._ria_matching(self._data_loader.get_test_document_paths(),
                                              self._data_loader.get_test_documents_texts(), sents, targ_vecs, targs)
        test_target_matches, test_target_matches_counts = self._data_loader.create_test_target_matches_dictionary()

        print('Performing RIA evaluation')
        self._matches_by_sent = self._evaluate_by_target(test_target_matches, test_target_matches_counts, 300)
        self._avg_matches_by_sent, self._avg_sdg_matches_by_sent = self._avg_matches(test_target_matches_counts, 300)

        print('Writing RIA results')
        results_filename = self._results_dir + 'RIA results - '
        if use_target_indicators:
            results_filename += 'use indicators - '
        if use_training_set_matches:
            results_filename += 'use training set matches - '
        if tfidf and not target_tfidf:
            results_filename += 'corpus tfidf'
        elif tfidf and target_tfidf:
            results_filename += 'target tfidf'
        elif not tfidf:
            results_filename += 'corpus nbow'
        self._save_results(results_filename)
        self._print_per_sdg_comparison(results_filename, label)
        self._print_per_target_comparison(results_filename, label)

        # Generate the results spreadsheet
        ria_results = self._get_results(50)
        RIA._generate_spreadsheet(ria_results, results_filename, self._data_loader.get_sdg_target_definitions(),
                                  self._data_loader.get_original_sdg_target_definitions())

        return self._matches_by_sent, self._avg_matches_by_sent

    def _get_target_embeddings(self):
        """
            Get target document embeddings and inverse dictionary.

            Returns:
                targs (dict[str]), targ_vecs(list[numpy.ndarray]), sents(list[str]) : inverse dictionary of target
                 documents (target document -> target_id), list of embedded target documents vectors, list of target
                 documents
        """
        targs = {}
        targ_vecs = []
        sents = []
        for key, val in self._target_docs.items():
            sents.append(str(val))
            targ_vecs.append(self._w2v.infer_vector(str(val)))
            targs[str(val)] = key
        return targs, targ_vecs, sents

    def _ria_matching(self, test_documents_paths, original_test_documents_texts, sents, targ_vecs, targs):
        """
            Find the sentences/paragaraphs of policy documents that most match each target.

            Given the (test) policy documents for a country that we wish to produce the RIA for, we will compare the
            similarity of each sentence/paragraph with the sentences from the target documents. Those sentences with the
            highest cosine similarity will be marked as matching the target of a particular target document.

        Args:
            test_documents_paths (list[string]) : list of paths to all test documents
            original_test_documents_texts ()    : the original textual contents of all test documents
            sents (list[str])                   : list of target documents
            targ_vecs (list[numpy.ndarray])     : list of embedded target documents vectors
            targs (dict[str])                   : inverse dictionary of target documents
        Returns:
            score_dict (dict[str]) : dictionary of target to tuples containing sentences found that match the target.
                Each tuple contains the cosine similarity score, the matched (processed) text, its original,
                unprocessed form and the surrounding context, as well document and text line information.
        """
        score_dict = {}
        for file in test_documents_paths:
            with open(file, 'r', encoding='utf-8') as test_doc_file:
                line_cnt = 0
                for line in test_doc_file:
                    line_cnt += 1
                    line_parts = line.strip().split('\t')
                    for i in range(0, len(line_parts)):
                        part = DataLoader.filter_text(line_parts[i])
                        if part != '':
                            top_matches = self._w2v.get_most_similar(sents, targ_vecs, part,
                                                                     RIA.NUMBER_OF_TARGETS - 1, 0.4)
                            for match in top_matches:
                                key = targs[match[1]]
                                filename = file[file.rfind('/')+1:]  # For excel output
                                original_line_parts = original_test_documents_texts[file][line_cnt-1].split('\t')
                                original_part = original_line_parts[i]
                                original_context = original_test_documents_texts[file][line_cnt-1]
                                if line_cnt > 1:
                                    original_context = original_test_documents_texts[file][line_cnt-2] + '\n' +\
                                                       original_context
                                if line_cnt < len(original_test_documents_texts[file]):
                                    original_context += ('\n' + original_test_documents_texts[file][line_cnt])
                                if key in score_dict:
                                    score_dict[key].add((match[0], (part, filename, line_cnt, original_part,
                                                                    original_context)))
                                else:
                                    score_dict[key] = {(match[0], (part, filename, line_cnt, original_part,
                                                                   original_context))}
        return score_dict

    def _get_matches(self, target_id, num_matches=300, get_source_info=False):
        """
            Returns the specified number of matches for a target in the target dictionary, ordered by cosine similarity.

        Args:
            target_id (str)       : Target to return matches for
            num_matches (int)     : Number of matches to be returned
            get_source_info(bool) : Whether to return only the sentence matches themselves, or a set of information for
                each match
        Returns:
            (list) : List of num_matches sentences/paragraphs that correspond to the specified target.
             If get_source_info is True return type is a list of tuples; otherwise it is a list of strings.
        """
        if get_source_info:
            ordered = [(str(item[0]), item[1][1], str(item[1][2]), item[1][0], item[1][3], item[1][4])
                       for item in sorted(self._score_dict[target_id], reverse=True)]
        else:
            ordered = [item[1][0] for item in sorted(self._score_dict[target_id], reverse=True)]
        return ordered[:num_matches]

    def _evaluate_by_target(self, test_target_matches, test_target_matches_counts, num):
        """
            Finds matches for all targets as the number of output sentences increases.

        Args:
            test_target_matches (dict[str]) : a dictionary of target : dict[sentence], where each sentence is in test
                set and was manually identified to match the target, and each dict[sentence] contains a mapping from
                sentence : number of its occurrences in the test set.
            test_target_matches_counts (dict[str]) : a dictionary of target : number of target matches in the test set
            num (int) : Number of output sentences to match to each target
        Returns:
            (dict[str]) : a dictionary of how many matches (in %) per target were found after each sentence
        """
        match_by_sent = {}
        true_matches_dict = {}

        for target in self._score_dict:
            for sentence in self._get_matches(target, num):
                if target in test_target_matches:
                    for true_match, true_match_count in test_target_matches[target].items():
                        score = SequenceMatcher(None, true_match, sentence).ratio()
                        if score > 0.9:
                            if target in true_matches_dict and\
                                            true_matches_dict[target].count(true_match) < true_match_count:
                                true_matches_dict[target].append(true_match)
                            elif target not in true_matches_dict:
                                true_matches_dict[target] = [true_match]

                    # Calculate how many true matches were found through AutoRIA, in percentages per target
                    if target in true_matches_dict:
                        if target in match_by_sent:
                            match_by_sent[target].append(
                                len(true_matches_dict[target]) / test_target_matches_counts[target])
                        else:
                            match_by_sent[target] = \
                                [len(true_matches_dict[target]) / test_target_matches_counts[target]]
                    else:
                        if target in match_by_sent:
                            match_by_sent[target].append(0)
                        else:
                            match_by_sent[target] = [0]
        return match_by_sent

    def _avg_matches(self, test_target_matches_counts, num):
        """
            Finds the average percent matches for all targets as the number of output sentences increases.
            Also finds the averaged matches for each SDG separately.

        Args:
            test_target_matches_counts (dict[str]) : a dictionary of target : number of target matches in the test set
            num (int) : Number of output sentences to match

        Returns:
            avg_total, avg_sdgs (list[float], dict[int]) : avg_total is the global AutoRIA performance list, in which
                the element at i-th index represents the overall algorithm performance (in terms of the % of detected
                matched sentences/paragraphs) with i output sentences per target. avg_sdgs is a dictionary containing
                similar performance lists for each SDG separately.
        """
        avg_total = []
        avg_sdgs = {}
        for i in range(1, 6):
            avg_sdgs[i] = []
        for i in range(num):
            adder, counter = 0, 0
            adder_sdgs = [0, 0, 0, 0, 0]
            counter_sdgs = [0, 0, 0, 0, 0]
            for key in self._matches_by_sent:
                try:
                    adder += (self._matches_by_sent[key][i] * test_target_matches_counts[key])
                    counter += test_target_matches_counts[key]
                    adder_sdgs[int(key[0])-1] += (self._matches_by_sent[key][i] * test_target_matches_counts[key])
                    counter_sdgs[int(key[0])-1] += test_target_matches_counts[key]
                except:
                    adder += (self._matches_by_sent[key][-1] * test_target_matches_counts[key])
                    counter += test_target_matches_counts[key]
                    adder_sdgs[int(key[0])-1] += (self._matches_by_sent[key][-1] * test_target_matches_counts[key])
                    counter_sdgs[int(key[0])-1] += test_target_matches_counts[key]
            avg_total.append(adder / counter)
            for j in range(1, 6):
                avg_sdgs[j].append(adder_sdgs[j-1]/counter_sdgs[j-1])
        return avg_total, avg_sdgs

    def _get_results(self, num_matches):
        """
            Retrieve the top num_matches most similar sentences/paragraphs for each SDG target

        Args:
            num_matches (int)  : Number of matches to be returned per target
        Returns:
            results (dict[str]) : Dictionary containing the matches for each target
        """
        results = {}
        for key in self._score_dict:
            results[key] = self._get_matches(key, num_matches, get_source_info=True)
        return results

    @staticmethod
    def _generate_spreadsheet(results, name, target_defs, original_target_defs):
        """
            Generate an excel spreadsheet of the results in which each sheet corresponds to a target.
            The first column is left blank for evaluation, the second column is the similarity score, the third is the
            origin document, the fourth the line of that document in which the target match is found, the fifth contains
            the matched text, while the sixth and seventh contain the original, unprocessed version of that text and its
            context.

        Args:
            results (dict) : target/sentence matches dictionary
            name (string)  : Spreadsheet name
            target_defs (dict[str]): processed SDG target definitions
            original_target_defs (dict[str]): unprocessed SDG target definitions
        """
        # Create a Pandas dataframe from the data.
        df = pd.DataFrame.from_dict(results, orient='index')
        x = df.transpose()

        xbook = xlsxwriter.Workbook(name + ' - predictions.xlsx')
        header = ['Manual assessment', 'Similarity score', 'Document', 'Document line', 'Matched text',
                  'Original text', 'Original context']

        # Convert the dataframe to an XlsxWriter Excel object.
        for col in sorted(x):
            xsheet = xbook.add_worksheet(str(col))
            xsheet.write_row(0, 0, [target_defs[col]])
            xsheet.write_row(1, 0, [original_target_defs[col]])
            xsheet.write_row(3, 0, header)
            for i in range(len(x[col])):
                xsheet.write_row(4+i, 1, list(x[col].loc[i]))
        xbook.close()

    def _save_results(self, results_filename):
        """
            Save the averaged performances on all targets for this RIA model, depending on the number of output
            sentences for each target.
            Performances are saved at several key/indicative points.

        Args:
            results_filename (str): String denoting the path and basic name for the file in which the results
                will be saved
        """
        with open(results_filename + '.txt', 'w', encoding='utf-8') as res_file:
            res_file.write('AVG performance (in %) on X sentences outputted per target\n')
            res_file.write('X = 10: ' + str(100 * self._avg_matches_by_sent[9]) + '\n')
            res_file.write('X = 30: ' + str(100 * self._avg_matches_by_sent[29]) + '\n')
            res_file.write('X = 50: ' + str(100 * self._avg_matches_by_sent[49]) + '\n')
            res_file.write('X = 100: ' + str(100 * self._avg_matches_by_sent[99]) + '\n')
            res_file.write('X = 200: ' + str(100 * self._avg_matches_by_sent[199]) + '\n')
            res_file.write('X = 300: ' + str(100 * self._avg_matches_by_sent[299]) + '\n')

    def _print_per_sdg_comparison(self, results_filename, label):
        """
            Print a RIA settings performance comparison chart for the selected 5 SDGs

        Args:
            results_filename (str): String denoting the path and basic name for the file in which the produced chart
                will be saved
            label (str): Description of the RIA setting which produced the performance plotted on the chart
        """
        sns.set_context('talk')
        sns.set_style("white")
        plt.figure(figsize=(15, 11))
        for key in range(1, 6):
            plt.plot(list(range(1, 101)), (np.asarray(self._avg_sdg_matches_by_sent[key]) * 100)[:100],
                     label='SDG ' + str(key))
        plt.plot(list(range(1, 101)), (np.asarray(self._avg_matches_by_sent) * 100)[:100], label='SDG Avg')
        plt.legend(title='SDG', bbox_to_anchor=(1.1, 1.2), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences by SDG - ' + label)
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 105, 10))
        plt.savefig(results_filename + ' - SDG comparison.jpg')
        plt.close()

    def _print_per_target_comparison(self, results_filename, label):
        """
            Print a RIA settings performance comparison chart for the selected SDG targets

        Args:
            results_filename (str): String denoting the path and basic name for the file in which the produced chart
                will be saved
            label (str): Description of the RIA setting which produced the performance plotted on the chart
        """
        sns.set_context('talk')
        sns.set_style("white")
        plt.figure(figsize=(15, 11))
        examples = ['1.4', '2.4', '3.8', '4.1', '5.2']
        for key in examples:
            plt.plot(list(range(1, 101)), (np.asarray(self._matches_by_sent[key]) * 100)[:100], label=key)
        plt.legend(title='SDG Target', bbox_to_anchor=(1.1, 1.2), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences by Target - ' + label)
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 105, 10))
        plt.savefig(results_filename + ' - target comparison.jpg')
        plt.close()
