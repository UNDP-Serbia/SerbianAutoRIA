# SerbianAutoRIA - a model for automating the Rapid Integrated Assessment (RIA) mechanism for Serbian
This model is based on the [previous IBM approach](https://github.com/IBM/Semantic-Search-for-Sustainable-Development) developed for English.
The model searches the test documents for sentences/paragraphs that are a semantic match for one the UN's Sustainable Development Goal (SDG) targets.

## RIA in Serbian
This repository was created within a UNDP pilot project for automating RIA in Serbian, a morphologically complex language with limited availability of natural language processing tools and resources.
The stemmer for Croatian by Ljubešić and Pandžić from the [SCStemmers](https://github.com/vukbatanovic/SCStemmers) package was used to process the textual data in this repository.
The pilot project evaluation focused on the first five SDGs.

## Running the RIA experiments
Should you wish to run the RIA experiments for Serbian, you may do so by calling the `experimenter.py` module, after downloading the provided code base and data.
Bear in mind that the SerbianAutoRIA model depends on the [gensim package](https://radimrehurek.com/gensim/) and its word2vec implementation.
Therefore, if you wish to avoid the effects of the initial word2vec network randomization, be sure to utilize only one processor thread and to set the PYTHONHASHSEED environment variable to a fixed value.
Otherwise, each run of the Experimenter will produce slightly different results.

## References
If you wish to use Serbian AutoRIA in your paper or project, please cite the following paper:

* **[Using Language Technologies to Automate the UNDP Rapid Integrated Assessment Mechanism in Serbian](http://lt4all.org/media/papers/O5/137.pdf)**, Vuk Batanović, Boško Nikolić, in Proceedings of the International Conference on Language Technologies for All: Enabling Linguistic Diversity and Multilingualism Worldwide (LT4All), Paris, France (2019).

The Serbian AutoRIA project is also described in the following blog post:
* [Natural Language Processing to align national plans in Serbia with Global Goals](https://undg.org/silofighters_blog/natural-language-processing-to-align-national-plans-in-serbia-with-global-goals/)

## Contact info
* Vuk Batanović - Innovation Center of the School of Electrical Engineering, University of Belgrade, Serbia - vuk.batanovic / at / ic.etf.bg.ac.rs
