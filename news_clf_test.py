import operator
import pickle
from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
with open('news.txt', encoding='utf-8') as f:
    text = [f.read()]
# create the transform
with open('CV.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
# encode document
vector = vectorizer.transform(text)
# test
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
lol = clf.predict_proba(vector)
print(lol)
print(lol[0][0])
print(lol[0][1])
print(clf.classes_)
