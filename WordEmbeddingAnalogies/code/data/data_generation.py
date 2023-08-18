import string
import wikipedia
import numpy as np
import nltk
from nltk.corpus import stopwords, gutenberg
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class GenData():

    def create_corpus(self, data):  # data cleansing

        stop_words = ['k', 'm', 't', 'd', 'e', 'f', 'g', 'h', 'i', 'u', 'r', 'I', 'im', 'ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this',
                      'should', 'our', 'their', 'while', 'above', 'both', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'would', 'could']
        filtered_sentences = []  # sentences without stop_words
        for i in range(len(data)):
            sentence = data[i]
            # temp = sentence.split()

            filtered_words = []
            for w in sentence:
                if w not in stop_words:
                    filtered_words.append(w)
            filtered_sentences.append(filtered_words)

        corpus = []  # filtered sentences minus sentences<2
        for i, sentence in enumerate(filtered_sentences):
            if (len(sentence) < 2):
                continue
            else:
                corpus.append(sentence)

        return corpus

    def lowercase(self, corpus):
        sentences_lower = []

        for sentence in corpus:
            sentences_lower.append([w.lower() for w in sentence])

        return sentences_lower

    def remove_stopwords(self, corpus):
        no_stopwords = []
        RemoveWords = stopwords.words('english')
        for sentence in corpus:
            no_stopwords.append([w for w in sentence if w not in RemoveWords])

        return no_stopwords

    def remove_punctuation(self, corpus):
        no_punctuation = []
        punctuation = string.punctuation
        for sentence in corpus:
            no_punctuation.append([word.translate(str.maketrans("", "", punctuation))
                                   for word in sentence if word.translate(str.maketrans("", "", punctuation)) != ""])

        return no_punctuation

    def apply_stemming(self, corpus):
        stemmer = PorterStemmer()
        stems = []

        for sentence in corpus:
            stems.append([stemmer.stem(w) for w in sentence])

        return stems

    def apply_lemmatization(self, corpus):
        lemmatizer = WordNetLemmatizer()
        lems = []

        for sentence in corpus:
            lems.append([lemmatizer.lemmatize(w) for w in sentence])

        return lems


if __name__ == '__main__':
    np.random.seed(0)

    model = GenData()

    data = gutenberg.sents('austen-emma.txt')
    data = list(data)

    with open('Analogy_dataset.txt', 'r') as file:
        words = set()
        for line in file:
            a, b, c, d = line.split()
            words.add(a)
            words.add(b)
            words.add(c)
            words.add(d)
    for word in words:
        try:
            wiki = wikipedia.page(word)
        except:
            continue

        text = wiki.content

        # Replace '==' with '' (an empty string)
        text = text.replace('==', '')

        # Replace '\n' (a new line) with '' & end the string at $1000.
        text = text.replace('\n', '')
        # text = text[0: 10000]
        # print(len(data))
        sentences = text.split('.')
        for sentence in sentences:
            data.append([i if i.isalpha() else 'a' for i in sentence.split()])
        print(len(text))


    corpus = model.create_corpus(data)
    corpus = model.lowercase(corpus)
    corpus = model.remove_stopwords(corpus)
    corpus = model.remove_punctuation(corpus)
    # corpus = model.apply_lemmatization(corpus)
    corpus = model.create_corpus(corpus)

    for sent in corpus: 
        for word in sent:
            with open("data_2.txt", "a") as myfile:
                myfile.write(word)
                myfile.write(" ")

