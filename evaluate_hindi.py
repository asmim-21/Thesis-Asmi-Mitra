import pandas as pd
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import nltk
from collections import Counter
import transformers
import re

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
transformers.logging.set_verbosity_error()

# Load data from CSV
def load_data(llm_name):
    path = f"results/{llm_name}_responses_hindi.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path)

# WEAT Score analysis using English gendered words
def weat_score(responses, female_words, male_words, neutral_words, model):
    embeddings = {word: model.encode(word) for word in female_words + male_words + neutral_words}
    scores = []
    for word in neutral_words:
        sim_female = np.mean([util.cos_sim(embeddings[word], embeddings[f])[0][0].item() for f in female_words])
        sim_male = np.mean([util.cos_sim(embeddings[word], embeddings[m])[0][0].item() for m in male_words])
        scores.append(sim_female - sim_male)
    return np.mean(scores)

# Sentiment analysis using BERT Model
def sentiment_analysis(responses):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    sentiments = sentiment_analyzer(responses)
    return pd.DataFrame(sentiments)

# Lexical diversity and frequency analysis
def lexical_stats(responses):
    tokens = nltk.word_tokenize(" ".join(responses).lower())
    diversity = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
    freq = Counter(tokens)
    return diversity, freq.most_common(20)

# Function to detect gendered verb conjugates in Hindi
def gendered_verb_conjugates(text):
    # Regular expressions for detecting verb conjugations
    male_conjugates = re.findall(r'\b(\w+ा|\w+े)\b', text)  # Male conjugates ending in 'ा' or 'े'
    female_conjugates = re.findall(r'\b(\w+ी|\w+ीं)\b', text)  # Female conjugates ending in 'ी' or 'ीं'
    
    return len(male_conjugates), len(female_conjugates)

# Bias determination based on frequencies of verb conjugates
def determine_bias(fm, ff):
    if fm > ff:
        return "Male biased"
    elif fm < ff:
        return "Female biased"
    else:
        return "Gender neutral"

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_hindi.py <llm_name>")
        sys.exit(1)

    llm_name = sys.argv[1]
    df = load_data(llm_name)
    responses = df['Response'].dropna().tolist()
    
    print("🔍 Running WEAT...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Multilingual model supporting Hindi
    female_words = ["महिला", "लड़की", "माँ", "बेटी", "वह", "उसकी"]  # Hindi female words
    male_words = ["पुरुष", "लड़का", "पिता", "बेटा", "वह", "उसका"]  # Hindi male words
    neutral_words = ["नेता", "डॉक्टर", "इंजीनियर", "शिक्षक", "नर्स"]  # Hindi neutral words
    weat = weat_score(responses, female_words, male_words, neutral_words, model)

    print("📊 Running Sentiment Analysis...")
    sentiment_df = sentiment_analysis(responses)

    print("🧠 Running Lexical Analysis...")
    diversity, freq_words = lexical_stats(responses)

    print("🔍 Running Gender Bias Detection...")
    fm, ff = 0, 0
    for response in responses:
        male_count, female_count = gendered_verb_conjugates(response)
        fm += male_count
        ff += female_count
    
    bias = determine_bias(fm, ff)

    print(f"\n✅ Evaluation for {llm_name.upper()} (HINDI):")
    print(f"- WEAT Score: {weat:.4f}")
    print(f"- Lexical Diversity: {diversity:.4f}")
    print(f"- Top Words: {freq_words}")
    print(f"- Sentiment Distribution:\n{sentiment_df['label'].value_counts(normalize=True)}")
    print(f"- Gender Bias: {bias} (Male conjugates: {fm}, Female conjugates: {ff})")

    # Save results
    os.makedirs("results", exist_ok=True)
    summary = {
        "LLM Name": llm_name.upper(),
        "WEAT Score": round(weat, 4),
        "Lexical Diversity": round(diversity, 4),
        "Top Words": freq_words,
        "Sentiment Distribution": sentiment_df['label'].value_counts(normalize=True).to_dict(),
        "Gender Bias": bias,
        "Male Conjugates": fm,
        "Female Conjugates": ff
    }

    with open(f"results/{llm_name}_hindi_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    sentiment_df.to_csv(f"results/{llm_name}_hindi_sentiment_details.csv", index=False)

if __name__ == "__main__":
    main()
