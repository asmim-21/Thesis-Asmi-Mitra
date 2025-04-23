import pandas as pd
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk
from collections import Counter
import transformers

nltk.download('punkt')
nltk.download('punkt_tab')
transformers.logging.set_verbosity_error()

def load_data(llm_name):
    path = f"results/{llm_name}_responses_hindi.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path)

def sentiment_analysis(responses):
    classifier = pipeline("sentiment-analysis", model="rajpurkar/HindiBERT-sentiment")
    sentiments = classifier(responses)
    return pd.DataFrame(sentiments)

def lexical_stats(responses):
    tokens = nltk.word_tokenize(" ".join(responses).lower())
    diversity = len(set(tokens)) / len(tokens)
    freq = Counter(tokens)
    return diversity, freq.most_common(20)

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_hindi.py <llm_name>")
        sys.exit(1)

    llm_name = sys.argv[1]
    df = load_data(llm_name)
    responses = df['Response'].dropna().tolist()

    print("📊 Running Sentiment Analysis...")
    sentiment_df = sentiment_analysis(responses)

    print("🧠 Running Lexical Analysis...")
    diversity, freq_words = lexical_stats(responses)

    print(f"\n✅ Evaluation for {llm_name.upper()} (HINDI):")
    print(f"- Lexical Diversity: {diversity:.4f}")
    print(f"- Top Words: {freq_words}")
    print(f"- Sentiment Distribution:\n{sentiment_df['label'].value_counts(normalize=True)}")

    # Save results
    os.makedirs("results", exist_ok=True)
    summary = {
        "LLM Name": llm_name.upper(),
        "Lexical Diversity": round(diversity, 4),
        "Top Words": freq_words,
        "Sentiment Distribution": sentiment_df['label'].value_counts(normalize=True).to_dict()
    }

    with open(f"results/{llm_name}_hindi_summary.txt", "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    sentiment_df.to_csv(f"results/{llm_name}_hindi_sentiment_details.csv", index=False)

if __name__ == "__main__":
    main()
