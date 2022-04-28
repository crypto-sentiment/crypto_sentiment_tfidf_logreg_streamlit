# Cryptonews sentiment classification model

A simple cryptonews sentiment classification app.

The model (simple tf-idf & logreg) is trained with ~4500 news titles manually classiified into positive, neutral or negative.

<img src='static/img/btc_sentiment_streamlit_app.png' width=300>

Launching the app:

 - install requirements from `requirements.txt`
 - execute `streamlit run src/app.py`

To run this as a background process, you can do `(streamlit run src/app.py > streamlit.log 2>&1 &)`. This will also write all Stramlit logs including error logs to the file `streamlit.log`.
