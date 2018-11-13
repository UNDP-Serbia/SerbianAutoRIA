from ria import RIA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Experimenter:
    """
        Main class for running RIA experiments.
    """

    def __init__(self, sdg_target_definitions_path='../data/SDG target definitions.xlsx',
                 sdg_target_document_matches_path='../data/SDG target - document text matches.xlsx',
                 training_test_split_path='../data/Training and test set division.xlsx',
                 documents_path='../data/documents/',
                 results_path='../results/',
                 embedding_dimensionalities=[300, 500, 1000],
                 threads=10
                 ):
        self._results_path = results_path
        self._embedding_dimensionalities = embedding_dimensionalities
        self._stem_rias = []
        self._no_stem_rias = []
        for embedding_dimensionality in embedding_dimensionalities:
            self._no_stem_rias.append(RIA(sdg_target_definitions_path=sdg_target_definitions_path,
                                          sdg_target_document_matches_path=sdg_target_document_matches_path,
                                          training_test_split_path=training_test_split_path,
                                          documents_path=documents_path,
                                          results_path=results_path,
                                          stemming=False,
                                          embedding_dimensionality=embedding_dimensionality,
                                          threads=threads
                                          ))
            self._stem_rias.append(RIA(sdg_target_definitions_path=sdg_target_definitions_path,
                                       sdg_target_document_matches_path=sdg_target_document_matches_path,
                                       training_test_split_path=training_test_split_path,
                                       documents_path=documents_path,
                                       results_path=results_path,
                                       stemming=True,
                                       embedding_dimensionality=embedding_dimensionality,
                                       threads=threads
                                       ))
        self.results = {}

    def run_experiments(self):
        """
            Run the RIA experiments, and save and print their results
        """
        best_no_stem = {}
        best_stem = {}
        for ria in self._no_stem_rias:
            avg_matches_by_sent_array, labels = self._compare_ria_settings(ria)
            embedding_dims = str(ria.get_embedding_dims())
            ind = labels.index('TFIDF + TS')
            best_no_stem[embedding_dims] = (avg_matches_by_sent_array[ind], embedding_dims + ' ' + labels[ind])
        for ria in self._stem_rias:
            avg_matches_by_sent_array, labels = self._compare_ria_settings(ria)
            embedding_dims = str(ria.get_embedding_dims())
            ind = labels.index('TFIDF + TS')
            best_stem[embedding_dims] = (avg_matches_by_sent_array[ind], embedding_dims + ' Stem ' + labels[ind])
        for dim in self._embedding_dimensionalities:
            avg_matches_by_sent_array_temp, labels_temp = zip(*[best_no_stem[str(dim)],
                                                                best_stem[str(dim)]])
            self._print_no_stem_vs_stem_comparison_charts(avg_matches_by_sent_array_temp, labels_temp, str(dim))
        values, labels = zip(*list(best_stem.values()))
        self._print_embedding_dimension_comparison_charts(values, labels)

    def _compare_ria_settings(self, ria):
        """
            Explore different RIA settings and return the results.

        Returns:
            avg_matches_by_sent_collection, labels (tuple(list[list[float]], list[str])) : a tuple consisting of a
             collection of global performance lists and of a list of labels, one label per global performance list.
             One list element corresponds to one RIA setting.
        """
        print('--------------------------------------------------')
        print('Running RIA experiments: embedding size ' + str(ria.get_embedding_dims()) + ', stemming: '
              + ('yes' if ria.uses_stemming() else 'no'))
        matches_by_sent_collection = []
        avg_matches_by_sent_collection = []
        matches_by_sent, avg_matches_by_sent = ria.run_ria('NBOW',
                                                           use_target_indicators=False,
                                                           use_training_set_matches=False,
                                                           tfidf=False,
                                                           target_tfidf=False)
        matches_by_sent_collection.append(matches_by_sent)
        avg_matches_by_sent_collection.append(avg_matches_by_sent)
        matches_by_sent, avg_matches_by_sent = ria.run_ria('TFIDF',
                                                           use_target_indicators=False,
                                                           use_training_set_matches=False,
                                                           tfidf=True,
                                                           target_tfidf=False)
        matches_by_sent_collection.append(matches_by_sent)
        avg_matches_by_sent_collection.append(avg_matches_by_sent)
        matches_by_sent, avg_matches_by_sent = ria.run_ria('TFIDF + Ind',
                                                           use_target_indicators=True,
                                                           use_training_set_matches=False,
                                                           tfidf=True,
                                                           target_tfidf=False)
        matches_by_sent_collection.append(matches_by_sent)
        avg_matches_by_sent_collection.append(avg_matches_by_sent)
        matches_by_sent, avg_matches_by_sent = ria.run_ria('TFIDF + TS',
                                                           use_target_indicators=False,
                                                           use_training_set_matches=True,
                                                           tfidf=True,
                                                           target_tfidf=False)
        matches_by_sent_collection.append(matches_by_sent)
        avg_matches_by_sent_collection.append(avg_matches_by_sent)
        matches_by_sent, avg_matches_by_sent = ria.run_ria('Target TFIDF + TS',
                                                           use_target_indicators=False,
                                                           use_training_set_matches=True,
                                                           tfidf=True,
                                                           target_tfidf=True)
        matches_by_sent_collection.append(matches_by_sent)
        avg_matches_by_sent_collection.append(avg_matches_by_sent)
        print('--------------------------------------------------')
        print('Printing comparison charts')
        labels = ['NBOW', 'TFIDF', 'TFIDF + Ind', 'TFIDF + TS', 'Target TFIDF + TS']
        self._print_settings_comparison_charts(avg_matches_by_sent_collection, labels, ria.get_embedding_dims(),
                                               ria.uses_stemming())
        return avg_matches_by_sent_collection, labels

    def _print_settings_comparison_charts(self, avg_matches_by_sent_collection, labels, embedding_dims, stemming):
        """
            Print a set of RIA settings performance comparison charts, reporting global/average performance levels
            depending on the number of sentences produced as candidates for each SDG target. The charts focus on the
            rise in performances up to three key sentence counts: 30, 50, and 300 sentences per target.

        Args:
            avg_matches_by_sent_collection (list[list[float]]): A collection of global/average performance lists
            labels (list[str]): Identification of settings corresponding to the given global performance lists
        """
        sns.set_context('talk')
        sns.set_style("white")

        plot_title = 'Setting (' + str(embedding_dims) + ', ' + ('stemming)' if stemming else 'no stemming)')
        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 31)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:30], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.2), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 55, 5))
        plt.savefig(self._results_path + str(embedding_dims) + ('/stem' if stemming else '/no_stem')
                    + '/settings_comparison_30.jpg')
        plt.close()

        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 51)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:50], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.2), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 55, 5))
        plt.savefig(self._results_path + str(embedding_dims) + ('/stem' if stemming else '/no_stem')
                    + '/settings_comparison_50.jpg')
        plt.close()

        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 301)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:300], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.2), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 105, 10))
        plt.savefig(self._results_path + str(embedding_dims) + ('/stem' if stemming else '/no_stem')
                    + '/settings_comparison_300.jpg')
        plt.close()

    def _print_no_stem_vs_stem_comparison_charts(self, avg_matches_by_sent_collection, labels, title_label):
        """
            Print a set of comparison charts between models in which text stemming is used and those in which it is not,
            reporting global/average performance levels depending on the number of sentences produced as candidates for
            each SDG target. The charts focus on the rise in performances up to three key sentence counts: 30, 50, and
            300 sentences per target.

        Args:
            avg_matches_by_sent_collection (list[list[float]]): A collection of global/average performance lists
            labels (list[str]): Identification of settings corresponding to the given global performance lists
            title_label (str): Text that more closely identifies the settings being compared. Used here for embedding
                dimensionality.
        """
        sns.set_context('talk')
        sns.set_style("white")

        plot_title = 'No stemming vs stemming (' + title_label + ')'
        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 31)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:30], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 55, 5))
        plt.savefig(self._results_path + title_label + '/no_stem_vs_stem_comparison_30.jpg')
        plt.close()

        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 51)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:50], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 55, 5))
        plt.savefig(self._results_path + title_label + '/no_stem_vs_stem_comparison_50.jpg')
        plt.close()

        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 301)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:300], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 105, 10))
        plt.savefig(self._results_path + title_label + '/no_stem_vs_stem_comparison_300.jpg')
        plt.close()

    def _print_embedding_dimension_comparison_charts(self, avg_matches_by_sent_collection, labels):
        """
            Print a set of comparison charts between models which use differently sized embeddings, reporting
            global/average performance levels depending on the number of sentences produced as candidates for
            each SDG target. The charts focus on the rise in performances up to three key sentence counts:
            30, 50, and 300 sentences per target.

        Args:
            avg_matches_by_sent_collection (list[list[float]]):  A collection of global/average performance lists
            labels (list[str]): Identification of settings corresponding to the given global performance lists
        """
        sns.set_context('talk')
        sns.set_style("white")

        plot_title = 'Embedding dimension'
        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 31)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:30], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 55, 5))
        plt.savefig(self._results_path + 'embedding_dimension_comparison_30.jpg')
        plt.close()

        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 51)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:50], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 55, 5))
        plt.savefig(self._results_path + 'embedding_dimension_comparison_50.jpg')
        plt.close()

        plt.figure(figsize=(15, 11))
        for i in range(0, len(avg_matches_by_sent_collection)):
            plt.plot(list(range(1, 301)), (np.asarray(avg_matches_by_sent_collection[i]) * 100)[:300], label=labels[i])
        plt.legend(title=plot_title, bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=10)
        plt.title('Percent Matches Vs. Number of Sentences')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Percent Matches with Policy Experts')
        plt.yticks(np.arange(0, 105, 10))
        plt.savefig(self._results_path + 'embedding_dimension_comparison_300.jpg')
        plt.close()

experimenter = Experimenter(embedding_dimensionalities=[300, 500, 1000], threads=1)
experimenter.run_experiments()
