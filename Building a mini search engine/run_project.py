from textwrap import indent
from tqdm import tqdm
from preprocessor import Preprocessor
from indexer import Indexer
from collections import OrderedDict
from linkedlist import LinkedList,Node
import inspect as inspector
import sys
import argparse
import json
import time
import random
import flask
from flask import Flask
from flask import request
import hashlib
import copy

app = Flask(__name__)


class ProjectRunner:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.indexer = Indexer()

    def _merge(self, postings_list1, postings_list2, skipflag):
        """ Implementing the merge algorithm to merge 2 postings list at a time.
            While merging 2 postings list, preserving the maximum tf-idf value of a document.
        """
        if skipflag==0:
            merged_list = LinkedList()
            post1 = copy.deepcopy(postings_list1)
            post2 = copy.deepcopy(postings_list2)
            comparisons = 0
            if post1 is not None and post2 is not None:
                p1 = post1.start_node
                p2 = post2.start_node
                while p1 and p2:
                    comparisons+=1
                    if p1.value == p2.value:
                        newnode=Node(value=p1.value)
                        if p1.tfidf>p2.tfidf:
                            newnode.tfidf=p1.tfidf
                        else:
                            newnode.tfidf=p2.tfidf
                        merged_list.insert_at_end(newnode.value,0,newnode.tfidf)
                        p1 = p1.next
                        p2 = p2.next
                    elif p1.value < p2.value:
                        p1 = p1.next

                    else:
                        p2 = p2.next
            return merged_list,comparisons
        else:
            merged_list = LinkedList()
            post1 = copy.deepcopy(postings_list1)
            post2 = copy.deepcopy(postings_list2)
            comparisons = 0
            if post1 is not None and post2 is not None:
                p1 = post1.start_node
                p2 = post2.start_node
                while p1 and p2:
                    comparisons+=1
                    if p1.value == p2.value:
                        newnode=Node(value=p1.value)
                        if p1.tfidf>p2.tfidf:
                            newnode.tfidf=p1.tfidf
                        else:
                            newnode.tfidf=p2.tfidf
                        merged_list.insert_at_end(newnode.value,0,newnode.tfidf)
                        p1 = p1.next
                        p2 = p2.next
                    elif p1.value < p2.value:
                        if p1.skipnext is not None and p1.skipnext.value<p2.value:
                            while(p1.skipnext is not None and p1.skipnext.value<p2.value):
                                p1=p1.skipnext
                                comparisons+=1
                        else:
                            p1=p1.next
                    elif p2.value < p1.value:
                        if p2.skipnext is not None and p2.skipnext.value<p1.value:
                            while(p2.skipnext is not None and p2.skipnext.value<p1.value):
                                p2=p2.skipnext
                                comparisons+=1
                        else:
                            p2=p2.next
            merged_list.add_skip_connections()
            return merged_list, comparisons

    def _daat(self, inputterms, skipflag):
        """ Implemented the DAAT AND algorithm, which merges the postings list of N query terms.
            Used appropriate parameters & return types.
        """
        if skipflag==0:
            merged_list = None
            total_comparisons = 0
            query_token_list = inputterms
            if len(query_token_list) == 1:
                merged_list = self.indexer.get_index()[query_token_list[0]]
            else:
                for i in range(1, len(query_token_list)):
                    if merged_list:
                        merged_list, comparisons = self._merge(merged_list, self.indexer.get_index()[query_token_list[i]],0)
                        total_comparisons += comparisons
                    else:
                        merged_list, comparisons = self._merge(self.indexer.get_index()[query_token_list[i-1]],self.indexer.get_index()[query_token_list[i]],0)
                        total_comparisons += comparisons
            merged_list_final=[]
            merged_list_final_tfidf=[]
            tempnode=merged_list.start_node
            while(tempnode):
                merged_list_final.append(tempnode.value)
                merged_list_final_tfidf.append((tempnode.tfidf,tempnode.value))
                tempnode=tempnode.next
            lst = len(merged_list_final_tfidf) 
            for i in range(0, lst):  
                for j in range(0, lst-i-1): 
                    if (merged_list_final_tfidf[j][0] < merged_list_final_tfidf[j + 1][0]): 
                        temp = merged_list_final_tfidf[j] 
                        merged_list_final_tfidf[j]= merged_list_final_tfidf[j + 1] 
                        merged_list_final_tfidf[j + 1]= temp 
            tfidf_daat=[]
            for i,j in merged_list_final_tfidf:
                tfidf_daat.append(j)
            return sorted(merged_list_final), tfidf_daat, total_comparisons

        else:
            merged_list = None
            total_comparisons = 0
            query_token_list = inputterms
            if len(query_token_list) == 1:
                merged_list = self.indexer.get_index()[query_token_list[0]]
            else:
                for i in range(1, len(query_token_list)):
                    if merged_list: 
                        merged_list, comparisons = self._merge(merged_list, self.indexer.get_index()[query_token_list[i]],1)
                        total_comparisons += comparisons
                    else:
                        merged_list, comparisons = self._merge(self.indexer.get_index()[query_token_list[i-1]],self.indexer.get_index()[query_token_list[i]],1)
                        total_comparisons += comparisons
            merged_list_final=[]
            merged_list_final_tfidf=[]
            tempnode=merged_list.start_node
            while(tempnode):
                merged_list_final.append(tempnode.value)
                merged_list_final_tfidf.append((tempnode.tfidf,tempnode.value))
                tempnode=tempnode.next
            lst = len(merged_list_final_tfidf) 
            for i in range(0, lst):  
                for j in range(0, lst-i-1): 
                    if (merged_list_final_tfidf[j][0] < merged_list_final_tfidf[j + 1][0]): 
                        temp = merged_list_final_tfidf[j] 
                        merged_list_final_tfidf[j]= merged_list_final_tfidf[j + 1] 
                        merged_list_final_tfidf[j + 1]= temp 
            tfidf_daat_skip=[]
            for i,j in merged_list_final_tfidf:
                tfidf_daat_skip.append(j)
            return sorted(merged_list_final), tfidf_daat_skip, total_comparisons

    def _get_postings(self,term):
        """ Function to get the postings list of a term from the index.
        """
        postingslist = self.indexer.get_index()[term].traverse_list()
        skiplist = self.indexer.get_index()[term].traverse_skips()
        return postingslist, skiplist

    def _output_formatter(self, op):
        """ This formats the result in the required format.
        """
        if op is None or len(op) == 0:
            return [], 0
        op_no_score = [int(i) for i in op]
        results_cnt = len(op_no_score)
        return op_no_score, results_cnt

    def run_indexer(self, corpus):
        """ This function reads & indexes the corpus. After creating the inverted index,
            it sorts the index by the terms, add skip pointers, and calculates the tf-idf scores.
        """
        with open(corpus, 'r') as fp:
            for line in tqdm(fp.readlines()):
                doc_id, document = self.preprocessor.get_doc_id(line)
                tokenized_document = self.preprocessor.tokenizer(document)
                self.indexer.generate_inverted_index(doc_id, tokenized_document)
        self.indexer.sort_terms()
        self.indexer.add_skip_connections()
        self.indexer.calculate_tf_idf()

    def sanity_checker(self, command):

        index = self.indexer.get_index()
        kw = random.choice(list(index.keys()))
        return {"index_type": str(type(index)),
                "indexer_type": str(type(self.indexer)),
                "post_mem": str(index[kw]),
                "post_type": str(type(index[kw])),
                "node_mem": str(index[kw].start_node),
                "node_type": str(type(index[kw].start_node)),
                "node_value": str(index[kw].start_node.value),
                "command_result": eval(command) if "." in command else ""}

    def run_queries(self, query_list, random_command):
        output_dict = {'postingsList': {},
                       'postingsListSkip': {},
                       'daatAnd': {},
                       'daatAndSkip': {},
                       'daatAndTfIdf': {},
                       'daatAndSkipTfIdf': {},
                       'sanity': self.sanity_checker(random_command)}

        for query in tqdm(query_list):
            """ Run each query against the index.
                1. Pre-process & tokenize the query.
                2. For each query token, get the postings list & postings list with skip pointers.
                3. Get the DAAT AND query results & number of comparisons with & without skip pointers.
                4. Get the DAAT AND query results & number of comparisons with & without skip pointers, 
                    along with sorting by tf-idf scores."""

            input_term_arr = self.preprocessor.tokenizer(query)
            input_term_arr=list(set(input_term_arr))
            sortedbylength=[]
            for term in input_term_arr:
                tup=(self.indexer.get_index()[term].length, term)
                sortedbylength.append(tup)
            sortedbylength.sort(key=lambda y: y[0])
            input_term_arr=[]
            for len,term in sortedbylength:
                input_term_arr.append(term)

            for term in input_term_arr:
                postings, skip_postings = self._get_postings(term)

                output_dict['postingsList'][term] = postings
                output_dict['postingsListSkip'][term] = skip_postings

            and_op_no_skip, and_op_no_skip_sorted, and_comparisons_no_skip = self._daat(input_term_arr,0)
            and_comparisons_no_skip_sorted = and_comparisons_no_skip
            and_op_skip, and_op_skip_sorted, and_comparisons_skip = self._daat(input_term_arr,1)
            and_comparisons_skip_sorted = and_comparisons_skip
            """ Implement logic to populate initialize the above variables.
                The below code formats the result to the required format.
            """
            and_op_no_score_no_skip, and_results_cnt_no_skip = self._output_formatter(and_op_no_skip)
            and_op_no_score_skip, and_results_cnt_skip = self._output_formatter(and_op_skip)
            and_op_no_score_no_skip_sorted, and_results_cnt_no_skip_sorted = self._output_formatter(and_op_no_skip_sorted)
            and_op_no_score_skip_sorted, and_results_cnt_skip_sorted = self._output_formatter(and_op_skip_sorted)

            output_dict['daatAnd'][query.strip()] = {}
            output_dict['daatAnd'][query.strip()]['results'] = and_op_no_score_no_skip
            output_dict['daatAnd'][query.strip()]['num_docs'] = and_results_cnt_no_skip
            output_dict['daatAnd'][query.strip()]['num_comparisons'] = and_comparisons_no_skip

            output_dict['daatAndSkip'][query.strip()] = {}
            output_dict['daatAndSkip'][query.strip()]['results'] = and_op_no_score_skip
            output_dict['daatAndSkip'][query.strip()]['num_docs'] = and_results_cnt_skip
            output_dict['daatAndSkip'][query.strip()]['num_comparisons'] = and_comparisons_skip

            output_dict['daatAndTfIdf'][query.strip()] = {}
            output_dict['daatAndTfIdf'][query.strip()]['results'] = and_op_no_score_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_docs'] = and_results_cnt_no_skip_sorted
            output_dict['daatAndTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_no_skip_sorted

            output_dict['daatAndSkipTfIdf'][query.strip()] = {}
            output_dict['daatAndSkipTfIdf'][query.strip()]['results'] = and_op_no_score_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_docs'] = and_results_cnt_skip_sorted
            output_dict['daatAndSkipTfIdf'][query.strip()]['num_comparisons'] = and_comparisons_skip_sorted

        return output_dict


@app.route("/execute_query", methods=['POST'])
def execute_query():
    """ This function handles the POST request to your endpoint.
    """
    start_time = time.time()

    queries = request.json["queries"]
    random_command = request.json["random_command"]

    """ Running the queries against the pre-loaded index. """
    output_dict = runner.run_queries(queries, random_command)

    """ Dumping the results to a JSON file. """
    with open(output_location, 'w') as fp:
        json.dump(output_dict, fp)

    response = {
        "Response": output_dict,
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    return flask.jsonify(response)



if __name__ == "__main__":
    """ Driver code for the project, which defines the global variables.
    """
    output_location = "project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, help="Corpus File name, with path.", default='/project/data/input_corpus.txt')
    parser.add_argument("--output_location", type=str, help="Output file name.", default='/project/data/output.json')
    parser.add_argument("--username", type=str,
                        help="Add a name you want",default='abhishek')

    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()
    
    """ Initialize the project runner"""
    runner = ProjectRunner()

    """ Indexed the documents from beforehand. When the API endpoint is hit, queries are run against 
        this pre-loaded in memory index. """
    runner.run_indexer(corpus)

    app.run(host="0.0.0.0", port=9999)
