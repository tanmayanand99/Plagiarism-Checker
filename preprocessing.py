import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer=PorterStemmer()

def removeUnnecessary(text):
    text = text.lower()
    text = re.sub("[^A-Za-z ]","",text)
    return text
def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    text1 = [word for word in text if word not in stop_words]
    return text1
def porterStemmer(text):
    text1 = [stemmer.stem(word) for word in text]
    return text1
def applyMyFuncs(text):
    return porterStemmer(removeStopWords(removeUnnecessary(text)))

    