import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import collections
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
import string


class Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        """ Splits each line of the document, into doc_id & text.
        """
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        """ Logic to pre-process & tokenize document text.
            This code can be re-used for processing the user's query.
        """
        letters_num=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','0']
        text_lower=text.lower()
        plaintext=''
        for i in text_lower:
            if i in letters_num:
                plaintext+=i
            else:
                plaintext+=' '
        text_l_r_strip=plaintext.strip()
        nospace_text=re.sub(' +',' ',text_l_r_strip)
        white_space_tokenized = nospace_text.split()
        no_stopwords= []
        for word in white_space_tokenized:
            if word not in self.stop_words:
                no_stopwords.append(word)
        stemmed_words=[]
        for word in no_stopwords:
            stemmed=self.ps.stem(word)
            stemmed_words.append(stemmed)
        
        return stemmed_words