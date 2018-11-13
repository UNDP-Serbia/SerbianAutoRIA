import os
import cyrtranslit


def transliterate_documents(doc_collection_path='../data/documents'):
    """
    Transliterates the contents of .txt files in the given directory path and saves each transliterated file
    to a .lat file.

    Args:
        doc_collection_path (str): The path of the directory containing the .txt documents
    Returns:
    """
    file_paths = os.listdir(doc_collection_path)
    file_paths = [doc_collection_path + '/' + doc_path for doc_path in file_paths if doc_path.endswith('.txt')]
    for input_path in file_paths:
        output_path = input_path + '.lat'
        with open(input_path, 'r', encoding='utf-8') as input_file:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    lat_line = cyrtranslit.to_latin(line)
                    output_file.write(lat_line.lower())

transliterate_documents()
