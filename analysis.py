import textstat
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

def get_readability_scores(text):
    return {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "SMOG Index": textstat.smog_index(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
    }

def detect_passive_voice(text):
    doc = nlp(text)
    passive_sentences = sum(1 for token in doc if "pass" in token.dep_)
    total_sentences = len(list(doc.sents))
    return round((passive_sentences / total_sentences) * 100, 2) if total_sentences else 0

def estimate_reading_time(text, wpm=200):
    words = text.split()
    return round(len(words) / wpm, 2)

def lexical_diversity(text):
    words = text.split()
    return round(len(set(words)) / len(words), 2) if words else 0

def get_sentiment(text):
    blob = TextBlob(text)
    return {
        "Polarity": blob.sentiment.polarity,
        "Subjectivity": blob.sentiment.subjectivity,
    }

def analyze_text(text):
    return {
        **get_readability_scores(text),
        "Passive Voice %": detect_passive_voice(text),
        "Reading Time (min)": estimate_reading_time(text),
        "Lexical Diversity": lexical_diversity(text),
        **get_sentiment(text)
    }
