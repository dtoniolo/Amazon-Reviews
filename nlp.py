import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
import string


stop_words = set(nltk.corpus.stopwords.words('english'))
wnl = nltk.stem.WordNetLemmatizer()


# NLP functions
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts

    Parameters
    ----------
    word : str
        Word to tag.

    Returns
    -------
    tag : str
        The word neg tag.

    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, nltk.wordnet.NOUN)


def tokenize(docs):
    """Tokenize a list of documents using nlkt word_tokenize.

    Parameters
    ----------
    docs : list of string
        List of documents to tokenize.

    Returns
    -------
    tokenized_docs : list of list of strings
        List of tokenized document. Each document is a list of tokens.

    """
    print('Tokenizing:')
    tokenized_docs = list()
    for doc in tqdm(docs):
        tokenized_docs.append(nltk.tokenize.word_tokenize(doc))
    return tokenized_docs


def remove_stopwords(docs):
    """Removes stop-words from a corpus of documents using nltk's stopwords
    list.

    Parameters
    ----------
    docs : list of list of string
        List of documents to modify. Each document is a list of tokens.

    Returns
    -------
    modified_docs : list of list of strings
        List of documents with stop words removed. Each document is a list of
        tokens.

    """
    print('Removing stop words:')
    global stop_words
    modified_docs = list()
    for doc in tqdm(docs):
        modified_docs.append(list())
        for token in doc:
            if token not in stop_words:
                modified_docs[-1].append(token)


def lemmatize(docs):
    """Lemmatize a corpus of documents using the WordNet lemmatizer.

    Parameters
    ----------
    docs : list of list of string
        List of documents to modify. Each document is a list of tokens.

    Returns
    -------
    lemmatized_docs : list of list of strings
        List of lemmatized documents removed. Each document is a list of
        tokens.

    """
    print('Lemmatizing:')
    global wnl
    lemmatized_docs = list()
    for doc in docs:
        lemmatized_docs.append(list())
        for token in doc:
            pos = get_wordnet_pos(token)
            lemmatized_docs[-1].append(wnl.lemmatize(token, pos))
    return lemmatized_docs


def remove_punctuation(docs):
    """Removes pure punctuation tokens from docs. And retransform each document
    back to a string.

    Parameters
    ----------
    docs : list of list of string
        List of documents to modify. Each document is a list of tokens.

    Returns
    -------
    modified_docs : list of list of strings
        List of documents with punctuation removed. Each document is a list of
        tokens.

    """
    print('Removing punctuation:')
    modified_docs = list()
    for doc in tqdm(docs):
        modified_docs.append(list())
        for token in doc:
            if token not in string.punctuation:
                modified_docs[-1].append(token)
        modified_docs[-1] = ' '.join(modified_docs[-1])
    return modified_docs
