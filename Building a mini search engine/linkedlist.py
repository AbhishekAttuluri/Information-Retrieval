import math

class Node:

    def __init__(self, value=None, next=None):
        """ Class to define the structure of each node in a linked list (postings list).
            Value: document id, Next: Pointer to the next node
            Added more parameters that are needed.
        """
        self.value = value
        self.tfidf = 0
        self.occurence = 0
        self.totalw = 0 
        self.skipnext = None
        self.next = next


class LinkedList:
    """ Class to define a linked list (postings list). Each element in the linked list is of the type 'Node'
        Each term in the inverted index has an associated linked list object.
    """
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length, self.n_skips, self.idf = 0, 0, 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        if self.start_node is None:
            traversal=[]
        else:
            tempnode=self.start_node
            while tempnode:
                traversal.append(tempnode.value)
                tempnode=tempnode.next
            """ Logic to traverse the linked list. """
            return traversal

    def traverse_skips(self):
        traversal = []
        if self.start_node is None or self.length<=2:
            traversal=[]
        else:
            tempnode=self.start_node
            while tempnode:
                traversal.append(tempnode.value)
                tempnode=tempnode.skipnext
            """ Logic to traverse the linked list. """
        return traversal

    def add_skip_connections(self):
        if self.length>2:
            self.n_skips = math.floor(math.sqrt(self.length))
            if self.n_skips * self.n_skips == self.length:
                self.n_skips = self.n_skips - 1
            self.skip_length = int(round(math.sqrt(self.length), 0))
            tempnode1=self.start_node
            tempnode2=self.start_node
            for i in range(0,self.n_skips):
                n=self.skip_length
                while(n):
                    tempnode1=tempnode1.next
                    n-=1
                tempnode2.skipnext=tempnode1
                tempnode2=tempnode1
        """ Logic to add skip pointers to the linked list. 
            This function does not return anything.
        """
    def add_tf_idf_scores(self):
        self.idf=5000/self.length
        tempnode=self.start_node
        while(tempnode):
            tf = tempnode.occurence/tempnode.totalw
            tempnode.tfidf = self.idf * tf
            tempnode=tempnode.next

    def insert_at_end(self, value, doclength, tfidf=0):
        new_node = Node(value=value)
        new_node.totalw=doclength
        new_node.tfidf=tfidf
        n = self.start_node
        if self.start_node is None:
            new_node.occurence=1
            self.start_node = new_node
            self.end_node = new_node
            self.length=1
            return
        elif self.start_node.value > value:
            new_node.occurence=1
            self.start_node = new_node
            self.start_node.next = n
            self.length+=1
            return
        elif self.end_node.value < value:
            new_node.occurence=1
            self.end_node.next = new_node
            self.end_node = new_node
            self.length+=1
            return
        elif self.end_node.value == value:
            self.end_node.occurence+=1
        elif self.start_node.value == value:
            self.start_node.occurence+=1
        else:
            while n.value <= value < self.end_node.value and n.next is not None:
                if n.value==value:
                    n.occurence+=1
                    return
                n = n.next

            m = self.start_node
            while m.next != n and m.next is not None:
                m = m.next
            new_node.occurence=1
            m.next = new_node
            new_node.next = n
            self.length+=1
            return