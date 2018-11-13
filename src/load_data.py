import os
import re
import string
import pandas as pd


class DataLoader:
    """
        Loads the RIA data from provided files
    """

    serbian_and_english_letters = 'AaBbVvGgDdĐđEeŽžZzIiJjKkLlMmNnOoPpRrSsTtĆćUuFfHhCcČčŠšQqWwYyXx'
    non_alphabetic_pattern = re.compile('[^' + serbian_and_english_letters + string.whitespace + ']+')
    short_words_pattern = re.compile(r'\b\w\b')

    @staticmethod
    def filter_text(text):
        """
            Pre-process the text - remove punctuation and digits, remove one-letter words.

        Args:
            text (str): input text
        Returns:
            filtered_text (str): filtered text
        """
        filtered_text = re.sub(r'/', ' ', text.strip())
        filtered_text = re.sub(DataLoader.short_words_pattern, '', filtered_text.strip())
        filtered_text = re.sub(DataLoader.non_alphabetic_pattern, '', filtered_text.strip())
        return filtered_text.strip()

    def __init__(self, sdg_target_definitions_path, sdg_target_document_matches_path,
                 training_test_split_path, stemming=True):
        self._sdg_target_definitions_xlsx = sdg_target_definitions_path
        self._sdg_target_document_matches_xlsx = sdg_target_document_matches_path
        self._training_test_split_xlsx = training_test_split_path
        self._use_stemming = stemming

        self._sdg_target_definitions = {}
        self._sdg_target_indicators = {}
        self._training_doc_ids = []
        self._test_doc_ids = []
        self._sdg_target_document_matches = []
        self._test_document_paths = []
        self._test_documents_texts = {}

        self._load_sdg_target_definitions()
        self._load_training_test_split()
        self._load_sdg_target_document_matches()

    def _load_sdg_target_definitions(self):
        """
            Loads a target ID : definition dictionary for SDG targets from an XLSX file.
        """
        sheet = pd.read_excel(self._sdg_target_definitions_xlsx, header=0)
        for index, row in sheet.iterrows():
            target_id = str(row['SDG target']).strip()
            if self._use_stemming:
                target_definition = row['Stemmed target']
                target_indicator = row['Stemmed indicator']
            else:
                target_definition = row['Full target']
                target_indicator = row['Full indicator']
            self._sdg_target_definitions[target_id] = DataLoader.filter_text(target_definition)
            self._sdg_target_indicators[target_id] = DataLoader.filter_text(target_indicator)

    def get_sdg_target_definitions(self):
        """
            Return the dictionary of stored, processed SDG target definitions.
            Dictionary is indexed using SDG target IDs, eg. '1.1'.

        Args:
        Returns:
            sdg_target_definitions (dict[str]): processed SDG target definitions
        """
        return self._sdg_target_definitions

    def get_original_sdg_target_definitions(self):
        """
            Return the dictionary of original, unprocessed SDG target definition texts.
            Dictionary is indexed using SDG target IDs, eg. '1.1'.

        Args:
        Returns:
            target_definitions (dict[str]): unprocessed SDG target definitions
        """
        sheet = pd.read_excel(self._sdg_target_definitions_xlsx, header=0)
        target_defs = {}
        for index, row in sheet.iterrows():
            target_id = str(row['SDG target']).strip()
            target_defs[target_id] = row['Full target'].strip()
        return target_defs

    def _load_training_test_split(self):
        """
            Loads a list of training and a list of test documents from an XLSX file.
        """
        sheet = pd.read_excel(self._training_test_split_xlsx, header=0)
        for index, row in sheet.iterrows():
            document_id = str(row['Document number']).strip()
            document_set = row['Training/test split'].strip()
            if document_set == 'Training':
                self._training_doc_ids.append(document_id)
            elif document_set == 'Test':
                self._test_doc_ids.append(document_id)

    def _load_sdg_target_document_matches(self):
        """
            Loads from an XLSX file a list of tuples containing SDG target IDs, document IDs, texts of the manually
            matched sentences and a boolean indicating if the document is in the training set.
        """
        sheet = pd.read_excel(self._sdg_target_document_matches_xlsx, header=0)
        for index, row in sheet.iterrows():
            target_id = str(row['SDG target']).strip()
            document_id = str(row['Document number']).strip()
            if self._use_stemming:
                sentence_match = row['Stemmed sentence']
            else:
                sentence_match = row['Sentence']
            if document_id in self._training_doc_ids:
                is_training_doc = True
            else:
                is_training_doc = False
            sentence_match = DataLoader.filter_text(sentence_match)
            self._sdg_target_document_matches.append((target_id, document_id, sentence_match, is_training_doc))

    def create_target_dictionary(self, use_target_indicators=True, use_training_set_matches=True):
        """
            Creates a dictionary of target : list[sentences]. The sentence list includes target definitions, by default.
            In addition, the sentence list can include target indicators, as well as sentences from the training set
            that were manually identified to match the target.
            Dictionary is indexed using SDG target IDs, eg. '1.1'.

            Args:
                use_target_indicators (bool): If true, target dictionary will contain the text of target indicator(s).
                use_training_set_matches (bool) : If true, target dictionary will contain sentences from the training
                 set that match the target.
            Returns:
                target_dict (dict[str]): Dictionary of target:list[sentences]
        """
        target_dict = {}
        for target_id, target_definition in self._sdg_target_definitions.items():
            target_dict[target_id] = [target_definition]
            if use_target_indicators:
                target_dict[target_id].append(self._sdg_target_indicators[target_id])
        if use_training_set_matches:
            for target_id, _, sentence, is_training_doc in self._sdg_target_document_matches:
                if is_training_doc:
                    target_dict[target_id].append(sentence)
        return target_dict

    def create_test_target_matches_dictionary(self):
        """
            Creates a dictionary of target : dict[sentence], where each sentence is in test set and was manually
            identified to match the target, and each dict[sentence] contains a mapping from sentence : number of its
            occurrences in the test set.
            Also creates a dictionary of target : number of target matches in the test set.
            Dictionaries are indexed using SDG target IDs, eg. '1.1'. Each subdictionary is indexed using the text of a
            particular sentence matched to the target.

            Args:
            Returns:
                (test_target_matches (dict[str]), test_target_matches_counts (dict[str])): a tuple of two dictionaries
        """
        test_target_matches = {}
        test_target_matches_counts = {}
        for target_id, _, sentence, is_training_doc in self._sdg_target_document_matches:
            if not is_training_doc:
                if target_id in test_target_matches:
                    if sentence in test_target_matches[target_id]:
                        test_target_matches[target_id][sentence] += 1
                    else:
                        test_target_matches[target_id][sentence] = 1
                    test_target_matches_counts[target_id] += 1
                else:
                    test_target_matches[target_id] = {}
                    test_target_matches[target_id][sentence] = 1
                    test_target_matches_counts[target_id] = 1
        return test_target_matches, test_target_matches_counts

    def create_target_documents(self, use_target_indicators=True, use_training_set_matches=True):
        """
            Creates a dictionary of target : target document, by merging into one document all sentences matching
            a target in the target dictionary. This includes target definitions, by default. In addition, the
            target document can include target indicators, as well as sentences from the training set that were
            manually identified to match the target.
            Dictionary is indexed using SDG target IDs, eg. '1.1'.

            Args:
                use_target_indicators (bool): If true, target documents will contain the text of target indicator(s).
                use_training_set_matches (bool) : If true, target documents will contain sentences from the training set
                    that match the target.
            Returns:
                target_documents (dict[str]) : Dictionary of target : target document
        """
        target_dict = self.create_target_dictionary(use_target_indicators, use_training_set_matches)
        target_documents = {}
        for key, val in target_dict.items():
            target_documents[key] = ' '.join(val)
        return target_documents

    def read_corpus(self, path='../data/documents/'):
        """
            Parse through every .lat or .stem file (depending on whether stemming is used) in the specified path.
            For every line in each file, filter the text. Yield a list of tokens for every processed line.

            Args:
                path (str) : Directory of the text documents (../data/documents/ by default)

            Yields:
                list[str] : a list of tokens from the filtered line
        """
        for file in os.listdir(path):
            document_id = file.split(' - ')[0]
            document_path = os.path.join(path, file)
            if file[-4:] == '.lat' and not self._use_stemming or file[-5:] == '.stem' and self._use_stemming:
                if document_id in self._test_doc_ids:
                    self._test_document_paths.append(document_path)
                    if not self._use_stemming:
                        self._test_documents_texts[document_path] = []
                with open(document_path, 'r', encoding='utf-8') as document_text:
                    for line in document_text:
                        if document_id in self._test_doc_ids and not self._use_stemming:
                            self._test_documents_texts[document_path].append(line.strip())
                        words = DataLoader.filter_text(line).split()
                        yield words
            elif file[-4:] == '.lat' and self._use_stemming and document_id in self._test_doc_ids:
                self._test_documents_texts[document_path+'.stem'] = []
                with open(document_path, 'r', encoding='utf-8') as document_text:
                    for line in document_text:
                        self._test_documents_texts[document_path+'.stem'].append(line.strip())

    def get_test_document_paths(self):
        """
            Get the list of paths to the test documents.

            Args:
            Returns:
                test_document_paths (list[str]) : list of paths to the test documents
        """
        return self._test_document_paths

    def get_test_documents_texts(self):
        """
            Get the dictionary of target : test document texts. The dictionary is indexed using the file path to a test
            document. The dictionary returns a list of strings, representing the text lines of the given test document.

            Args:
            Returns:
                test_documents_texts(dict[str]) - dictionary of test document path : test document text
        """
        return self._test_documents_texts
