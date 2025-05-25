import pandas as pd
import re

# Simulated positive/negative word lists and stopwords
positive_words = set(["good", "great", "excellent", "positive", "fortunate", "correct", "superior"])
negative_words = set(["bad", "poor", "wrong", "negative", "inferior", "unfortunate"])
custom_stopwords = set(["the", "is", "in", "a", "we", "and", "has", "been", "to", "of", "for", "on", "that"])

# Simulated articles
sample_articles = {
    '101': "The economy is in a good state. We see great improvements and a positive trend overall.",
    '102': "Unfortunately, the weather has been bad. The negative impact is visible in daily life."
}

def clean_text(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()

def tokenize_sentences(text):
    return re.split(r'[.!?]', text)

def tokenize_words(text):
    return re.findall(r'\b\w+\b', text)

def get_sentiment_scores(tokens):
    pos = sum(1 for word in tokens if word in positive_words)
    neg = sum(1 for word in tokens if word in negative_words)
    polarity = (pos - neg) / ((pos + neg) + 0.000001)
    subjectivity = (pos + neg) / (len(tokens) + 0.000001)
    return pos, neg, polarity, subjectivity

def count_syllables(word):
    count = len(re.findall(r'[aeiouy]+', word.lower()))
    if word.endswith(("es", "ed")):
        count -= 1
    return max(count, 1)

def count_complex_words(words):
    return sum(1 for word in words if count_syllables(word) > 2)

def analyze_text(text):
    text = clean_text(text)
    sentences = [s for s in tokenize_sentences(text) if s.strip()]
    words = tokenize_words(text)
    words = [w for w in words if w not in custom_stopwords]

    wc = len(words)
    sc = len(sentences)
    avg_sl = wc / sc if sc else 0

    complex_wc = count_complex_words(words)
    perc_complex = complex_wc / wc if wc else 0
    fog = 0.4 * (avg_sl + perc_complex)

    syllables = sum(count_syllables(w) for w in words)
    syll_per_word = syllables / wc if wc else 0

    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))

    avg_wl = sum(len(w) for w in words) / wc if wc else 0

    pos, neg, polarity, subj = get_sentiment_scores(words)

    return [pos, neg, polarity, subj, avg_sl, perc_complex, fog, avg_sl,
            complex_wc, wc, syll_per_word, personal_pronouns, avg_wl]

# Input URLs
input_df = pd.DataFrame({
    'URL_ID': ['101', '102'],
    'URL': ['https://example.com/article1', 'https://example.com/article2']
})

# Analyze and collect results
results = []
for idx, row in input_df.iterrows():
    text = sample_articles.get(row['URL_ID'], "")
    analysis = analyze_text(text)
    results.append([row['URL_ID'], row['URL']] + analysis)

# Final output DataFrame
columns = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
    'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
    'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]

output_df = pd.DataFrame(results, columns=columns)

# Save to Excel (current directory)
output_path = 'output.xlsx'
output_df.to_excel(output_path, index=False)
print(f"Output saved to: {output_path}")
