from flask import Flask, render_template, request
from textblob import TextBlob
import re

app = Flask(__name__)


def polarityCheck(text_obj):
    if text_obj.sentiment.polarity > 0:
        return "positive"
    elif text_obj.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"


@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html", title="Welcome to Sentiment Classifier")


@app.route("/predictOpinion", methods=['POST'])
def predict():
    if request.method == 'POST':
        opinion = request.form["opinion"]

        #Preprocessing of the text to extract the sentences into a python list
        sentences = re.compile(r'''(?<=[.!?]['"\s])\s*(?=[A-Z])''').split(opinion)
        opinion = [element.strip() for element in sentences]

        net_polarity = 0
        final_analysis = []
        for sentence in opinion:
            text_obj = TextBlob(sentence)
            final_analysis.append((sentence, polarityCheck(text_obj)))
            net_polarity += text_obj.sentiment.polarity
        net_polarity /= len(opinion)
    return render_template("results.html", final_analysis=final_analysis, title="Prediction Results", net_polarity=net_polarity)

       
if __name__ == "__main__":
    app.run(debug=True)
