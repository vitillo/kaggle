import pandas as pd
import re
import nltk
import logging

from gensim import corpora
from bs4 import BeautifulSoup
from gensim.utils import revdict
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords_en = stopwords.words('english')


def extract_words(review):
    text = re.sub("[^a-z]", " ", review.lower())
    words = text.split()
    return [w for w in words if not w in stopwords_en]


def extract_sentence_list(review):
    text = BeautifulSoup(review).get_text()
    sentences = tokenizer.tokenize(text)
    return map(extract_words, sentences)


def extract_sentences(reviews):
    review_sentences = frame["review"].map(extract_sentence_list)
    sentences = [sentence for sentence_list in review_sentences for sentence in sentence_list]
    return sentences


def contexts(document, window):
    assert(window % 2 == 0)

    half_window = window/2

    for i in range(half_window, len(document) - window):
        yield document[i:i + half_window], document[i + half_window], document[i + half_window + 1: i + window + 1]


if __name__ == "__main__":
    frame = pd.read_csv("/Users/vitillo/Downloads/labeledTrainData.tsv", sep='\t')
    sentences = extract_sentences(frame["review"])
    dictionary = corpora.Dictionary(sentences)

    dictionary.filter_extremes(no_below=5, no_above=0.6, keep_n=1000)
    rev_dictionary = revdict(dictionary)

    with open('data.csv', 'w') as f:
        for sentence in sentences:
            document = [rev_dictionary[w] for w in sentence if w in rev_dictionary]

            for context in contexts(document, 4):
                prefix, word, postfix = context

                f.write(",".join(map(str, prefix + postfix + [word])))
                f.write("\n")
 
    dictionary.save_as_text("dictionary.tsv")
