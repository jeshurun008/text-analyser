import streamlit as st
from analysis import analyze_text

st.title(" How Smart Is Your Text?")

text = st.text_area("Paste your text here:", height=300)

if st.button("Analyze"):
    results = analyze_text(text)
    st.subheader("📊 Analysis Results")
    for key, value in results.items():
        st.write(f"**{key}:** {value}")
