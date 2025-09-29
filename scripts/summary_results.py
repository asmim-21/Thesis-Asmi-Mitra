
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
BASE_DIR = "results"
FIGURE_DIR = os.path.join("figures", "summary")
MODELS = ["gemini", "claude", "openai"]
LANGUAGES = ["english", "hindi"]
TECHNIQUES = ["explicit_instruction", "few_shot", "role_prompt"]

# Ensure output directory exists
os.makedirs(FIGURE_DIR, exist_ok=True)

# Data Loading Functions
def load_all_summary():
    """
    Load summary CSVs for all models, techniques, and languages.
    Returns a DataFrame with columns: model, technique, language, weat, gender_bias, lexical_diversity
    """
    records = []
    for model in MODELS:
        for technique in TECHNIQUES:
            for language in LANGUAGES:
                path = os.path.join(
                    BASE_DIR, model, "prompt_engineering",
                    f"{model}_{technique}_bias_{language}_summary.csv"
                )
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                row = df.iloc[0]
                records.append({
                    "model": model,
                    "technique": technique,
                    "language": language,
                    "weat": row["Average WEAT Score"],
                    "gender_bias": row["Overall Gender Bias"],
                    "lexical_diversity": row["Lexical Diversity"]
                })
    return pd.DataFrame(records)

# Loading response analysis data for all models, techniques, and languages
def load_all_responses():
    """
    Load response analysis CSVs for all models, techniques, and languages.
    Returns a DataFrame with columns: model, technique, language, word_limit, weat
    """
    records = []
    for model in MODELS:
        for technique in TECHNIQUES:
            for language in LANGUAGES:
                path = os.path.join(
                    BASE_DIR, model, "prompt_engineering",
                    f"{model}_{technique}_bias_{language}_response_analysis.csv"
                )
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                if "Word Limit" not in df.columns:
                    continue
                # Find the WEAT score column (case-insensitive)
                weat_col = None
                for col in df.columns:
                    if "weat" in col.lower():
                        weat_col = col
                        break
                if weat_col is None:
                    continue
                for _, row in df.iterrows():
                    records.append({
                        "model": model,
                        "technique": technique,
                        "language": language,
                        "word_limit": row["Word Limit"],
                        "weat": row[weat_col]
                    })
    return pd.DataFrame(records)

# Plotting heatmap of WEAT scores by model and language
def plot_heatmap(df):
    """
    Plot a heatmap of WEAT score by model and language - relevant for research question 1
    """
    pivot = df.groupby(["model", "language"]).mean(numeric_only=True).reset_index()
    heatmap_data = pivot.pivot(index="model", columns="language", values="weat")
    plt.figure(figsize=(6, 4))
    heatmap_data.index = [str(m).title() for m in heatmap_data.index]   # Format row and column labels for display
    heatmap_data.columns = [str(l).title() for l in heatmap_data.columns]

    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".5f")

    plt.title("Comparison of WEAT Bias Scores Across Models and Languages")
    plt.ylabel("Model")
    plt.xlabel("Language")
    plt.tight_layout()

    plt.savefig(os.path.join(FIGURE_DIR, "heatmap_model_language_weat.png"))
    plt.close()

# Plotting line graphs of WEAT score by technique, with subplots for each model and lines for each language
def plot_line_technique_language(df):
    """
    Plot line graphs of WEAT score by technique, with subplots for each model and lines for each language - relevant for research question 2
    """
    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 5), sharey=True)
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        ax.yaxis.set_visible(True)
        ax.tick_params(axis='y', which='both', labelleft=True)
        sub = df[df["model"] == model]

        for language in LANGUAGES:
            lang_sub = sub[sub["language"] == language]
            ax.plot(
                TECHNIQUES,
                [lang_sub[lang_sub["technique"] == t]["weat"].mean() for t in TECHNIQUES],
                marker="o",
                label=language.title()
            )

        ax.set_title(model.title())
        ax.set_xlabel("Prompt Engineering Technique")
        ax.set_xticks(range(len(TECHNIQUES)))
        ax.set_xticklabels([t.replace("_", " ").title() for t in TECHNIQUES])
        ax.set_ylabel("Average WEAT Score")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.suptitle("Effect of Prompt Engineering Technique on WEAT Bias Score for Each Model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "line_technique_language.png"))
    plt.close()

# Plotting scatter plot of WEAT score vs word limit, with subplots for each language and hue for model
def plot_scatter_word_limit(df):
    """
    Plot a stripplot of WEAT score vs word limit, with subplots for each language and hue for model - relevant for research question 3
    """
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["word_limit", "weat", "language"])
    plot_df["word_limit"] = pd.to_numeric(plot_df["word_limit"], errors="coerce") # Convert to numeric (in case of string type)
    plot_df["weat"] = pd.to_numeric(plot_df["weat"], errors="coerce")
    plot_df = plot_df.dropna(subset=["word_limit", "weat"])
    plot_df = plot_df[plot_df["word_limit"].isin([50, 100, 200])]
    plot_df["word_limit"] = plot_df["word_limit"].astype(int).astype(str)  # Make word_limit categorical for y-axis
    plot_df["word_limit"] = pd.Categorical(plot_df["word_limit"], categories=["50", "100", "200"], ordered=True)

    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(12, 5), sharey=True)
    if len(LANGUAGES) == 1:
        axes = [axes]
    for idx, language in enumerate(LANGUAGES):
        ax = axes[idx]
        ax.yaxis.set_visible(True)
        ax.tick_params(axis='y', which='both', labelleft=True)
        lang_df = plot_df[plot_df["language"] == language]
        if lang_df.empty:
            continue
        sns.stripplot(
            data=lang_df,
            x="weat",
            y="word_limit",
            hue="model",
            alpha=0.6,
            dodge=True,
            jitter=True,
            ax=ax
        )
        ax.set_title(language.title())
        ax.set_xlabel("WEAT Score (Per Response)")
        ax.set_ylabel("Response Length")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["50", "100", "200"])
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.suptitle("Relationship Between WEAT Bias Score and Response Length by Model and Language")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "scatter_word_limit_weat.png"))
    plt.close()

# Plotting line graphs of WEAT score by technique, with subplots for each language and lines for each model
def plot_line_technique_model(df):
    """
    Plot line graphs of WEAT score by technique, with subplots for each language and lines for each model - relevant for research question 4
    """
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(12, 5), sharey=True)
    if len(LANGUAGES) == 1:
        axes = [axes]
    for idx, language in enumerate(LANGUAGES):
        ax = axes[idx]
        ax.yaxis.set_visible(True)
        ax.tick_params(axis='y', which='both', labelleft=True)
        sub = df[df["language"] == language]
        for model in MODELS:
            model_sub = sub[sub["model"] == model]
            ax.plot(
                TECHNIQUES,
                [model_sub[model_sub["technique"] == t]["weat"].mean() for t in TECHNIQUES],
                marker="o",
                label=model.title()
            )
        ax.set_title(language.title())
        ax.set_xlabel("Prompt Engineering Technique")
        ax.set_xticks(range(len(TECHNIQUES)))
        ax.set_xticklabels([t.replace("_", " ").title() for t in TECHNIQUES])
        ax.set_ylabel("Average WEAT Score")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.suptitle("Comparison of WEAT Bias Scores Across Models and Languages for Each Prompt Engineering Technique")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "line_technique_model.png"))
    plt.close()

# Main Execution
if __name__ == "__main__":
    # Load data
    summary_df = load_all_summary()
    response_df = load_all_responses()

    # Generate and save all summary plots
    plot_heatmap(summary_df)
    plot_line_technique_language(summary_df)
    plot_scatter_word_limit(response_df)
    plot_line_technique_model(summary_df)
    print("Summary plots saved to:", FIGURE_DIR)