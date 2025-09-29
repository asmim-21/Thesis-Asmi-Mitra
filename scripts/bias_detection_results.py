import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Seaborn Style
sns.set(style="whitegrid")

# Constants
BASE_DIR = "results"
LANGUAGES = ['english', 'hindi']
SENTIMENT_LABELS = ['5 stars', '4 stars', '3 stars', '2 stars', '1 star']
GENDER_COLORS = {'Male': 'blue', 'Female': 'pink', 'Neutral': 'green'}

# Word lists for frequency analysis in both languages
LANGUAGE_WORD_MAPS = {
    'english': {
        'female': ["woman", "girl", "mother", "daughter", "she", "her", "female", "sister", "aunt", "wife", "lady",
                   "girlhood", "feminine", "queen", "princess", "daughter-in-law", "housewife", "matriarch",
                   "bachelorette", "mom", "ladyboss", "herself", "hers", "her's"],
        'male': ["man", "boy", "father", "son", "he", "him", "male", "brother", "uncle", "husband", "gentleman",
                 "manhood", "masculine", "king", "prince", "brotherhood", "bachelor", "fatherhood", "patriarch",
                 "dad", "father-in-law", "dude", "sir", "guy", "mister", "himself", "his", "he's"],
        'neutral': ["leader", "doctor", "engineer", "teacher", "nurse", "scientist", "manager", "driver", "worker",
                    "student", "lawyer", "developer", "director", "coach", "principal", "CEO", "entrepreneur",
                    "author", "advisor", "technician", "artist", "musician", "journalist", "strategist",
                    "administrator", "architect", "consultant", "researcher", "professor"]
    },
    'hindi': {
        'female': ["महिला", "लड़की", "माँ", "बेटी", "वह", "उसकी", "स्त्री", "बहन", "आंटी", "पत्नी", "राजकुमारी",
                   "नारीत्व", "महिलावादी", "घरेलू महिला", "माँजी", "गृहणी", "विधवा", "सास", "कन्या", "मात्री शक्ति",
                   "स्वतंत्र महिला", "उनकी", "माँ की"],
        'male': ["पुरुष", "लड़का", "पिता", "बेटा", "वह", "उसका", "भाई", "चाचा", "पति", "राजा",
                 "राजकुमार", "संप्रभु", "पितृशक्ति", "शादीशुदा", "दादा", "साहब", "जेंटलमैन", "गाय",
                 "सखा"],
        'neutral': ["नेता", "डॉक्टर", "इंजीनियर", "शिक्षक", "नर्स", "वैज्ञानिक", "प्रबंधक", "ड्राइवर", "कर्मचारी",
                    "छात्रा", "वकील", "डेवलपर", "निर्देशक", "कोच", "प्रधान", "सीईओ", "उद्यमी", "लेखक", "सलाहकार",
                    "कलाकार", "संगीतज्ञ", "पत्रकार", "स्ट्रैटेजिस्ट", "प्रशासक", "आर्किटेक्ट", "कंसल्टेंट",
                    "अनुसंधानकर्ता", "प्रोफेसर", "टीचर", "व्यक्ति", "मैनेजर"]
    }
}

# Associating Word Lists to genders
def get_gender_tag(word, language):
    word_lower = word.lower()
    if word_lower in LANGUAGE_WORD_MAPS[language]['male']:
        return 'Male'
    elif word_lower in LANGUAGE_WORD_MAPS[language]['female']:
        return 'Female'
    else:
        return 'Neutral'

# Load data from CSV files of different models and languages
def load_summary_csv(model_name, language):
    path = os.path.join(BASE_DIR, model_name, "bias_detection", f"{model_name}_{language}_summary.csv")
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        'WEAT_Score': row['Average WEAT Score'],
        'Lexical_Diversity': row['Lexical Diversity'],
        'Top_Words': ast.literal_eval(row['Top Words']),
        'Sentiment_Distribution': ast.literal_eval(row['Sentiment Distribution']),
        'Male_Conjugates': row['Total Male Conjugates'],
        'Female_Conjugates': row['Total Female Conjugates'],
        'Gender_Bias': row['Overall Gender Bias']
    }

def load_response_csv(model_name, language):
    path = os.path.join(BASE_DIR, model_name, "bias_detection", f"{model_name}_{language}_response_analysis.csv")
    return pd.read_csv(path)

# Save plots to appropriate directories
def save_plot(title, model_name):
    print(title)
    filename = title.replace(" - ", "_") \
                    .replace("(", "") \
                    .replace(")", "") \
                    .replace(" ", "_") \
                    .lower() + ".png"
                    
    folder_path = os.path.join("figures", model_name, "bias_detection")
    os.makedirs(folder_path, exist_ok=True)
    path = os.path.join(folder_path, filename)
    plt.savefig(path)
    print(f"Saved: {path}")

# Plotting gender conjugate counts for each model and language
def plot_gender_conjugate_barchart(data, model_name, language):
    df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'count': [data.get('Male_Conjugates', 0), data.get('Female_Conjugates', 0)]
    })
    plt.figure(figsize=(6, 4))
    sns.barplot(x='gender', y='count', data=df, hue='gender', dodge=False, palette=GENDER_COLORS, width=0.4, legend=False)
    title = f"Gender Conjugate Counts - {model_name.title()} ({language.title()})"
    plt.title(title)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(title, model_name)
    plt.close()

# Plotting top word frequencies for each model and language
def plot_word_frequency_barchart(data, model_name, language):
    top_words = data.get('Top_Words', [])
    if not top_words:
        print(f"No Top Words data for {model_name} {language}")
        return
    df = pd.DataFrame(top_words, columns=['word', 'count'])
    df['gender'] = df['word'].apply(lambda w: get_gender_tag(w, language))
    df = df.sort_values('count', ascending=False).head(20).iloc[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(x='count', y='word', data=df, hue='gender', dodge=False, palette=GENDER_COLORS, width=0.4, legend=False)

    # Add custom legend
    legend_patches = [Patch(color=color, label=gender) for gender, color in GENDER_COLORS.items()]
    plt.legend(handles=legend_patches, title="Gender", loc='upper right')

    title = f"Top 20 Frequent Words - {model_name.title()} ({language.title()})"
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    save_plot(title, model_name)
    plt.close()

# Plotting sentiment distribution pie chart for each model and language
def plot_sentiment_distribution(df, model_name, language):
    if 'Sentiment Label' not in df.columns:
        print(f"Missing 'Sentiment Label' column for {model_name} {language}")
        return
    sentiment_counts = df['Sentiment Label'].value_counts().reindex(SENTIMENT_LABELS, fill_value=0)

    plt.figure(figsize=(6, 6))
    sentiment_counts.plot(
        kind='pie',
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=90,
        colors=['lightgreen', 'lightgray', 'lightcoral', 'lightyellow', 'lightblue']
    )
    title = f"Sentiment Distribution - {model_name.title()} ({language.title()})"
    plt.title(title)
    plt.ylabel('')
    plt.tight_layout()
    save_plot(title, model_name)
    plt.close()

# Generate all charts for a given model across languages
def generate_all_charts(model_name):
    for language in LANGUAGES:
        print(f"\nProcessing {model_name.title()} ({language.title()})")

        # Set font depending on language
        if language == 'hindi':
            plt.rcParams['font.family'] = 'Mangal'  
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'  

        try:
            bias_data = load_summary_csv(model_name, language)
            response_df = load_response_csv(model_name, language)

            plot_gender_conjugate_barchart(bias_data, model_name, language)
            plot_word_frequency_barchart(bias_data, model_name, language)
            plot_sentiment_distribution(response_df, model_name, language)
        except Exception as e:
            print(f"Error processing {model_name} {language}: {e}")

# Main Execution
if __name__ == "__main__":
    # Generate all charts for each model
    generate_all_charts("gemini")
    generate_all_charts("claude")
    generate_all_charts("openai")