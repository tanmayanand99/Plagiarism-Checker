from flask import Flask
from flask import render_template,request
from nltk.tokenize import sent_tokenize
import numpy as np
from preprocessing import removeUnnecessary,removeStopWords,porterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

nltk.download('punkt')
nltk.download('stopwords')

p = np.load('data.npy')
notprocessed = np.load('notprocessed.npy')

with open('saved_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def check():
    data = request.form['text']
    sent=sent_tokenize(data)
    npt = np.append(notprocessed,sent)
    for i in range(len(sent)):
        sent[i]=" ".join(porterStemmer(removeStopWords(removeUnnecessary(sent[i]).split(" "))))
    flen = len(p)
    k=np.append(p,sent)
    my_stop_words = stopwords.words('english')
    tf_idf_vect = TfidfVectorizer(stop_words=my_stop_words)
    X_train_tf_idf = tf_idf_vect.fit_transform(k)
    terms = tf_idf_vect.get_feature_names()
    final={}
    sum=0
    for i in range(flen,len(k)):
        dp = cosine_similarity(X_train_tf_idf[i],X_train_tf_idf[0:flen])
        m=np.argmax(dp)
        print("Simmilarity With:")
        sum=sum+(dp[0][m]*100)
        if(dp[0][m]>0.3):
            print(loaded_dict[npt[m]])
            final[npt[i]]=loaded_dict[npt[m]]
            print(dp[0][m])
        else:
            final[npt[i]]="0"
            print("No simmiliarity")
        print()
    sim=sum/(len(k)-flen)
    return render_template('main.html', data=final,sim=sim)

if __name__=="__main__":
    app.run(debug=True)