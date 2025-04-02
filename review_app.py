import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model = pk.load(open("model.pk1", "rb"))
scalar = pk.load(open("cv.pk2", "rb"))
review = st.text_input("Enter Movie Review ")

if st.button("Predict"):
    review_scale = scalar.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        st.write("Negative Review")
    else:
        st.write("Positive Review")    