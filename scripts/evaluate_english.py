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
nltk.download('averaged_perceptron_tagger_eng')
transformers.logging.set_verbosity_error()

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path)

# English WEAT Score Calculation (per response)
def weat_score(responses, model):
    # Defining the word lists within the function
    female_words = ["woman", "girl", "mother", "daughter", "she", "her", "female", "sister", "aunt", "wife", "lady", 
                    "girlhood", "feminine", "queen", "princess", "daughter-in-law", "housewife", "matriarch", 
                    "bachelorette", "mom", "ladyboss", "herself", "hers", "her's"]
    male_words = ["man", "boy", "father", "son", "he", "him", "male", "brother", "uncle", "husband", "gentleman", 
                  "manhood", "masculine", "king", "prince", "brotherhood", "bachelor", "fatherhood", "patriarch", 
                  "dad", "father-in-law", "dude", "sir", "guy", "mister", "himself", "his", "he's"]
    neutral_words = ["leader", "doctor", "engineer", "teacher", "nurse", "scientist", "manager", "driver", "worker", 
                     "student", "lawyer", "developer", "director", "coach", "principal", "CEO", "entrepreneur", 
                     "author", "advisor", "technician", "artist", "musician", "journalist", "actor", "scientist", 
                     "entrepreneur", "strategist", "administrator", "architect", "consultant", "researcher", "professor"]
    
    # Encoding the words from the word lists
    embeddings = {word: model.encode(word) for word in female_words + male_words + neutral_words}
    
    # Calculating the mean similarity for each response
    scores = []
    for response in responses:
        response_embedding = model.encode(response)
        
        # Calculating average similarity between the response and the female/male/neutral word sets
        sim_female = np.mean([util.cos_sim(response_embedding, embeddings[f])[0][0].item() for f in female_words])
        sim_male = np.mean([util.cos_sim(response_embedding, embeddings[m])[0][0].item() for m in male_words])
        sim_neutral = np.mean([util.cos_sim(response_embedding, embeddings[n])[0][0].item() for n in neutral_words])
        
        # Computing the difference (female - male) for this specific response
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
    text = " ".join(responses).lower()
    tokens = nltk.tokenize.word_tokenize(text)
    
    # Keeping only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha()]
    
    # POS tagging
    tagged_tokens = nltk.pos_tag(tokens)
    
    # Removing only function words  
    words_to_remove_pos = {'IN', 'DT', 'CC', 'TO', 'UH', 'RP'}
    
    filtered_tokens = [word for word, pos in tagged_tokens if pos not in words_to_remove_pos]

    diversity = len(set(filtered_tokens)) / len(filtered_tokens)
    freq = Counter(filtered_tokens)

    return diversity, freq.most_common(20)

# Function to detect gendered pronouns and corresponding verb conjugations
def gendered_verb_conjugates(text):
    # Defining pronouns associated with male and female references
    male_pronouns = ['he', 'him', 'his', 'himself', "he's"]
    female_pronouns = ['she', 'her', 'hers', 'herself', "she's"]
    
    # Tokenizing the text and get a list of words
    words = nltk.tokenize.word_tokenize(text.lower())
    
    male_conjugates = 0
    female_conjugates = 0
    
    # Iterating through the tokens and find pronouns
    for i, word in enumerate(words):
        if word in male_pronouns:
            # Checking the next word for a verb conjugation (e.g., "he plays", "he worked")
            if i + 1 < len(words) and re.match(r'\w+ed|\w+s|\w+ing', words[i + 1]):
                male_conjugates += 1
        elif word in female_pronouns:
            # Checking the next word for a verb conjugation (e.g., "she plays", "she worked")
            if i + 1 < len(words) and re.match(r'\w+ed|\w+s|\w+ing', words[i + 1]):
                female_conjugates += 1
                
    return male_conjugates, female_conjugates

# Bias determination based on frequencies of verb conjugates
def determine_bias(fm, ff):
    if fm > ff:
        return "Male biased"
    elif fm < ff:
        return "Female biased"
    else:
        return "Gender neutral"

def row_bias(m, f):
    if m > f:
        return "Male"
    elif f > m:
        return "Female"
    else:
        return "Neutral"
    
# Main function to run all analysis and save results
def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_english.py <input_csv_path> <output_name_prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_prefix = sys.argv[2]

    df = pd.read_csv(input_path)
    responses = df['Response'].dropna().tolist()
    prompts = df['Prompt'].tolist() if 'Prompt' in df.columns else [''] * len(responses)
    word_limits = df['WordLimit'].tolist() if 'WordLimit' in df.columns else [None] * len(responses)

    print("Running WEAT...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    weat_scores = weat_score(responses, model)

    print("Running Sentiment Analysis...")
    sentiments_raw = sentiment_analysis(responses)
    sentiments = [item['label'] for item in sentiments_raw]
    sentiment_scores = [float(item['score']) for item in sentiments_raw]

    print("Running Lexical Analysis...")
    diversity, freq_words = lexical_stats(responses)

    print("Running Gender Bias Detection...")
    fm, ff = 0, 0
    male_counts, female_counts, row_biases = [], [], []
    for r in responses:
        m, f = gendered_verb_conjugates(r)
        fm += m
        ff += f
        male_counts.append(m)
        female_counts.append(f)
        row_biases.append(row_bias(m, f))

    overall_bias = determine_bias(fm, ff)

    # Save response-level CSV
    response_df = pd.DataFrame({
        "Word Limit": word_limits,
        "Prompt": prompts,
        "Response": responses,
        "Sentiment Label": sentiments,
        "Sentiment Score": sentiment_scores,
        "WEAT Score": weat_scores,
        "Male Conjugates": male_counts,
        "Female Conjugates": female_counts,
        "Gender Bias": row_biases
    })
    response_df.to_csv(f"{output_prefix}_response_analysis.csv", index=False)

    # Save summary CSV
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
    pd.DataFrame([summary_data]).to_csv(f"{output_prefix}_summary.csv", index=False)

    print(f"\nDone! Files saved:\n- {output_prefix}_response_analysis.csv\n- {output_prefix}_summary.csv")

if __name__ == "__main__":
    main()