import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sns.set(style="whitegrid")

BASE_DIR = "results"
LANGUAGES = ['english', 'hindi']
SENTIMENT_LABELS = ['5 stars', '4 stars', '3 stars', '2 stars', '1 star']
TECHNIQUES = ['explicit_instruction', 'few_shot', 'role_prompt']
GENDER_COLORS = {'Male': 'blue', 'Female': 'pink', 'Neutral': 'green'}

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

def get_gender_tag(word, language):
    word_lower = word.lower()
    if word_lower in LANGUAGE_WORD_MAPS[language]['male']:
        return 'Male'
    elif word_lower in LANGUAGE_WORD_MAPS[language]['female']:
        return 'Female'
    else:
        return 'Neutral'

def load_summary_csv(model_name, technique, language):
    path = os.path.join(
        BASE_DIR, model_name, "prompt_engineering",
        f"{model_name}_{technique}_bias_{language}_summary.csv"
    )
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

def load_response_csv(model_name, technique, language):
    path = os.path.join(
        BASE_DIR, model_name, "prompt_engineering",
        f"{model_name}_{technique}_bias_{language}_response_analysis.csv"
    )
    return pd.read_csv(path)

def save_plot(title, model_name, technique):
    filename = title.replace(" - ", "_") \
                    .replace("(", "") \
                    .replace(")", "") \
                    .replace(" ", "_") \
                    .lower() + ".png"
                    
    folder_path = os.path.join("figures", model_name, "prompt_engineering", technique)
    os.makedirs(folder_path, exist_ok=True)
    path = os.path.join(folder_path, filename)
    plt.savefig(path)
    print(f"✅ Saved: {path}")

def plot_gender_conjugate_barchart(data, model_name, technique, language):
    df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'count': [data.get('Male_Conjugates', 0), data.get('Female_Conjugates', 0)]
    })
    plt.figure(figsize=(6, 4))
    sns.barplot(x='gender', y='count', data=df, hue='gender', dodge=False, palette=GENDER_COLORS, width=0.4, legend=False)
    title = f"Gender Conjugate Counts - {technique.title()} - {model_name.title()} ({language.title()})"
    plt.title(title)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(title, model_name, technique)
    plt.close()

def plot_word_frequency_barchart(data, model_name, technique, language):
    top_words = data.get('Top_Words', [])
    if not top_words:
        print(f"No Top Words data for {model_name} {technique} {language}")
        return
    df = pd.DataFrame(top_words, columns=['word', 'count'])
    df['gender'] = df['word'].apply(lambda w: get_gender_tag(w, language))
    df = df.sort_values('count', ascending=False).head(20).iloc[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(x='count', y='word', data=df, hue='gender', dodge=False, palette=GENDER_COLORS, width=0.4, legend=False)

    # Add custom legend
    legend_patches = [Patch(color=color, label=gender) for gender, color in GENDER_COLORS.items()]
    plt.legend(handles=legend_patches, title="Gender", loc='upper right')

    title = f"Top 20 Frequent Words - {technique.title()} - {model_name.title()} ({language.title()})"
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    save_plot(title, model_name, technique)
    plt.close()

def plot_sentiment_distribution(df, model_name, technique, language):
    if 'Sentiment Label' not in df.columns:
        print(f"Missing 'Sentiment Label' column for {model_name} {technique} {language}")
        return
    sentiment_counts = df['Sentiment Label'].value_counts().reindex(SENTIMENT_LABELS, fill_value=0)

    plt.figure(figsize=(6, 6))
    sentiment_counts.plot(
        kind='pie',
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=90,
        colors=['lightgreen', 'lightgray', 'lightcoral', 'lightyellow', 'lightblue']
    )
    title = f"Sentiment Distribution - {technique.title()} - {model_name.title()} ({language.title()})"
    plt.title(title)
    plt.ylabel('')
    plt.tight_layout()
    save_plot(title, model_name, technique)
    plt.close()

def generate_all_charts(model_name):
    for lang in LANGUAGES:
        print(f"\n📊 Processing {model_name.title()} ({lang.title()})")

        # Set font depending on language
        if lang == 'hindi':
            plt.rcParams['font.family'] = 'Mangal'  
        else:
            plt.rcParams['font.family'] = 'DejaVu Sans'  

        try:
            bias_data = load_summary_csv(model_name, lang)
            response_df = load_response_csv(model_name, lang)

            plot_gender_conjugate_barchart(bias_data, model_name, lang)
            plot_word_frequency_barchart(bias_data, model_name, lang)
            plot_sentiment_distribution(response_df, model_name, lang)
        except Exception as e:
            print(f"❌ Error processing {model_name} {lang}: {e}")


def generate_all_charts(model_name):
    for technique in TECHNIQUES:
        for language in LANGUAGES:
            print(f"\n📊 Processing {model_name} - {technique} - {language}")

            # Set font depending on language
            if language == 'hindi':
                plt.rcParams['font.family'] = 'Mangal'  
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'  

            try:
                bias_data = load_summary_csv(model_name, technique, language)
                response_df = load_response_csv(model_name, technique, language)

                plot_gender_conjugate_barchart(bias_data, model_name, technique, language)
                plot_word_frequency_barchart(bias_data, model_name, technique, language)
                plot_sentiment_distribution(response_df, model_name, technique, language)
            except Exception as e:
                print(f"❌ Error processing {model_name} - {technique} - {language}: {e}")


if __name__ == "__main__":
    generate_all_charts("gemini")
    generate_all_charts("claude")
    generate_all_charts("openai")