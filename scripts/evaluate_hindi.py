import re
import os
import sys
import nltk
import numpy as np
import pandas as pd
import transformers
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
transformers.logging.set_verbosity_error()

# Load CSV
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path)

# Hindi WEAT Score Calculation (per response)
def weat_score(responses, model):
    female_words = ["महिला", "लड़की", "माँ", "बेटी", "वह", "उसकी", "स्त्री", "बहन", "आंटी", "पत्नी", "राजकुमारी", 
                    "माँ", "नारीत्व", "महिलावादी", "राजकुमारी", "घरेलू महिला", "माँजी", "गृहणी", "विधवा", "सास", 
                    "कन्या", "मात्री शक्ति", "स्वतंत्र महिला", "उनकी", "माँ की"]
    
    male_words = ["पुरुष", "लड़का", "पिता", "बेटा", "वह", "उसका", "पुरुष", "भाई", "चाचा", "पति", "राजा", 
                  "राजकुमार", "पिता", "संप्रभु", "पति", "पितृशक्ति", "शादीशुदा", "दादा", "चाचा", "साहब", 
                  "जेंटलमैन", "लड़का", "गाय", "सखा", "उसका"]
    
    neutral_words = ["नेता", "डॉक्टर", "इंजीनियर", "शिक्षक", "नर्स", "वैज्ञानिक", "प्रबंधक", "ड्राइवर", "कर्मचारी", 
                     "छात्रा", "वकील", "डेवलपर", "निर्देशक", "कोच", "प्रधान", "सीईओ", "उद्यमी", "लेखक", "सलाहकार", 
                     "कलाकार", "संगीतज्ञ", "पत्रकार", "वैज्ञानिक", "उद्यमी", "स्ट्रैटेजिस्ट", "प्रशासक", "आर्किटेक्ट", 
                     "कंसल्टेंट", "अनुसंधानकर्ता", "प्रोफेसर", "टीचर", "व्यक्ति", "मैनेजर"]
    
    # Encode the words from the word lists
    embeddings = {word: model.encode(word) for word in female_words + male_words + neutral_words}
    
    # Calculate the mean similarity for each response
    scores = []
    for response in responses:
        response_embedding = model.encode(response)
        
        # Calculate average similarity between the response and the female/male/neutral word sets
        sim_female = np.mean([util.cos_sim(response_embedding, embeddings[f])[0][0].item() for f in female_words])
        sim_male = np.mean([util.cos_sim(response_embedding, embeddings[m])[0][0].item() for m in male_words])
        sim_neutral = np.mean([util.cos_sim(response_embedding, embeddings[n])[0][0].item() for n in neutral_words])
        
        # Compute the difference (female - male) for this specific response
        score = sim_female - sim_male
        scores.append(score)

    return scores

# Sentiment analysis using BERT Model
def sentiment_analysis(responses):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    sentiments = sentiment_analyzer(
        responses,
        truncation=True,
        max_length=512,
        return_all_scores=False
    )

    return sentiments

# Lexical diversity and frequency analysis
def lexical_stats(responses):
    # Join all responses into one string and tokenize
    text = " ".join(responses).lower()
    tokens = nltk.tokenize.word_tokenize(text)

    # Keep only alphabetic tokens (filter out punctuation and numbers)
    tokens = [t for t in tokens if re.match(r'^[\u0900-\u097F]+$', t)]  # Hindi alphabet Unicode range

    diversity = len(set(tokens)) / len(tokens) if tokens else 0
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

# Row-level bias
def row_bias(m, f):
    if m > f:
        return "Male"
    elif f > m:
        return "Female"
    else:
        return "Neutral"

# Main function to run all analyses and save results
def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_hindi.py <input_csv_path> <output_name_prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_name_prefix = sys.argv[2]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No file found at {input_path}")

    df = pd.read_csv(input_path)
    responses = df['Response'].dropna().tolist()
    prompts = df['Prompt'].tolist() if 'Prompt' in df.columns else [''] * len(responses)
    word_limits = df['WordLimit'].tolist() if 'WordLimit' in df.columns else [None] * len(responses)

    print("Running WEAT...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    weat_scores = weat_score(responses, model)

    print("Running Sentiment Analysis...")
    sentiment_results = sentiment_analysis(responses)
    sentiments = [item['label'] for item in sentiment_results]
    sentiment_scores = [float(item['score']) for item in sentiment_results]

    print("Running Lexical Analysis...")
    diversity, freq_words = lexical_stats(responses)

    print("Running Gender Bias Detection...")
    fm, ff = 0, 0
    male_conj_counts, female_conj_counts, bias_per_response = [], [], []

    for response in responses:
        m, f = gendered_verb_conjugates(response)
        fm += m
        ff += f
        male_conj_counts.append(m)
        female_conj_counts.append(f)
        bias_per_response.append(row_bias(m, f))

    overall_bias = determine_bias(fm, ff)

    # Response-level dataframe
    response_df = pd.DataFrame({
        "Word Limit": word_limits,
        "Prompt": prompts,
        "Response": responses,
        "Sentiment Label": sentiments,
        "Sentiment Score": sentiment_scores,
        "WEAT Score": weat_scores,
        "Male Conjugates": male_conj_counts,
        "Female Conjugates": female_conj_counts,
        "Gender Bias": bias_per_response
    })
    response_df.to_csv(f"{output_name_prefix}_response_analysis.csv", index=False)

    # Summary-level dataframe
    summary_data = {
        "Input File": input_path,
        "Average WEAT Score": round(np.mean(weat_scores), 4),
        "Lexical Diversity": round(diversity, 4),
        "Top Words": str(freq_words),
        "Sentiment Distribution": str(pd.Series(sentiments).value_counts(normalize=True).to_dict()),
        "Overall Gender Bias": overall_bias,
        "Total Male Conjugates": fm,
        "Total Female Conjugates": ff
    }
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(f"{output_name_prefix}_summary.csv", index=False)

    print(f"\nDone! Files saved:\n- {output_name_prefix}_response_analysis.csv\n- {output_name_prefix}_summary.csv")

if __name__ == "__main__":
    main()