from flask import Flask, request, render_template
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

set(stopwords.words('english'))
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    stop_words = stopwords.words('english')
    text1 = request.form['text1'].lower()  
    lang = TextBlob(text1)
    parse_lang = lang.detect_language()
    processed = ' '.join([word for word in text1.split() if word not in stop_words])
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('index.html', final=compound, text1=text1, processed=processed, lang=parse_lang)

# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=8081, debug=True)
    
#     app.run(host="0.0.0.0", port=80, debug=True, threaded=True)
if __name__ == "__main__": 
    app.run()