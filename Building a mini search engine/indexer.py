from linkedlist import LinkedList
from collections import OrderedDict


class Indexer:
    def __init__(self):
        """ Add more attributes if needed"""
        self.inverted_index = OrderedDict({})

    def get_index(self):
        """ Function to get the index. """
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        """ This function adds each tokenized document to the index. This in turn uses the function add_to_index """
        doclength=len(tokenized_document)
        for t in tokenized_document:
            self.add_to_index(t, doc_id, doclength)

    def add_to_index(self, term_, doc_id_, doclength):
        """ This function adds each term & document id to the index.
            If a term is not present in the index, then add the term to the index & initialize a new postings list (linked list).
            If a term is present, then add the document to the appropriate position in the postings list of the term.
        """
        if term_ in self.inverted_index:
            self.inverted_index[term_].insert_at_end(doc_id_,doclength)
        else:
            self.inverted_index[term_]= LinkedList()
            self.inverted_index[term_].insert_at_end(doc_id_,doclength)
        #raise NotImplementedError

    def sort_terms(self):
        """ Sorting the index by terms."""
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        """ For each postings list in the index, add skip pointers. """
        for key in self.inverted_index:
            self.inverted_index[key].add_skip_connections()

    def calculate_tf_idf(self):
        """ Calculate tf-idf score for each document in the postings lists of the index. """
        for key in self.inverted_index:
            self.inverted_index[key].add_tf_idf_scores()