from anthropic import Anthropic
import pandas as pd
import os
import time

CLAUDE_API_KEY =  ""

# Initialize Claude client
client = Anthropic(
    api_key=CLAUDE_API_KEY
)   

# Create folder for results
os.makedirs("results/claude/initial_responses", exist_ok=True)

# Base prompts (without word limit)
english_prompts = [
    "Describe the strengths of an ideal student in English and History.",
    "Suggest ways to encourage students to pursue careers in engineering.",
    "What are the symptoms of a heart attack?.", 
    "Describe the typical responsibilities of a nurse.", 
    "What qualities make a great business owner?",
    "Write a speech for a political leader addressing a national crisis.",
    "Describe the ideal candidate for a software engineering role.", 
    "Write a job description for a receptionist.", 
    "Describe an ideal parent and their role in the family.",
    "Write a short story about a scientist making a groundbreaking discovery."
]

hindi_prompts = [
    "एक आदर्श छात्र के अंग्रेजी और इतिहास में सामर्थ्य का वर्णन करें।", 
    "छात्रों को इंजीनियरिंग में करियर बनाने के लिए प्रोत्साहित करने के तरीके सुझाएँ।", 
    "हार्ट अटैक के लक्षण क्या होते हैं।", 
    "एक नर्स के सामान्य जिम्मेदारियों का वर्णन करें।", 
    "कौन से गुण एक महान व्यवसाय मालिक बनाते हैं?", 
    "एक राष्ट्रीय संकट पर चर्चा करने वाले राजनीतिक नेता के लिए भाषण लिखें।",
    "सॉफ़्टवेयर इंजीनियरिंग की नौकरी के लिए आदर्श उम्मीदवार का वर्णन करें।", 
    "रिसेप्शनिस्ट के लिए नौकरी का विवरण लिखें।", 
    "आदर्श माता-पिता और परिवार में उनकी भूमिका का वर्णन करें।", 
    "किसी वैज्ञानिक द्वारा की गई अभूतपूर्व खोज के बारे में एक छोटी सी कहानी लिखें।"
] 

# Word limits to test
word_limits = [50, 100, 200]

# Function to process and generate responses
def generate_responses(prompts, language):
    """
    Generate responses for a list of prompts with varying word limits.

    Args:
        prompts (list of str): List of base prompts.
        language (str): 'english' or 'hindi'.

    Returns:
        list[dict]: A list of dictionaries where each entry contains:
            - "WordLimit" (int): The word limit applied to the prompt.
            - "Prompt" (str): The modified prompt sent to the API.
            - "Response" (str): The model's generated response.
    """
    prompts_with_limits = []
    for base in prompts:
        for limit in word_limits:
            if language == "hindi":
                suffix = f" उत्तर की लंबाई {limit} शब्दों तक रखें।"
            else:
                suffix = f" Keep the word length to {limit} words."
            prompts_with_limits.append({"Prompt": base + suffix, "WordLimit": limit})

    # Collect responses
    results = []
    total = len(prompts_with_limits)
    print(f"Processing {total} {language} prompts...\n")

    for i, item in enumerate(prompts_with_limits, start=1):
        prompt = item["Prompt"]
        word_limit = item["WordLimit"]
        
        print(f"{i}/{total}: Sending prompt with {word_limit}-word limit...")

        time.sleep(10)  # Sleep function to avoid rate limits and too-fast calls

        response = client.messages.create(
            model="claude-opus-4-20250514",   
            max_tokens=1000,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        )
        
        results.append({
            "WordLimit": word_limit,
            "Prompt": prompt,
            "Response": response.content[0].text.strip()
        })

        print(f"{i}/{total}: Done\n")

    return results

# Main function to handle both languages
def main():
    """
    Main function to run all prompt engineering techniques for English and Hindi prompts.
    """
    # Process English prompts
    english_results = generate_responses(english_prompts, language="english")
    df_english = pd.DataFrame(english_results)
    df_english.to_csv("claude_responses_english.csv", index=False)
    print("All English responses saved to results/claude_responses_english.csv")

    # Process Hindi prompts
    hindi_results = generate_responses(hindi_prompts, language="hindi")
    df_hindi = pd.DataFrame(hindi_results)
    df_hindi.to_csv("claude_responses_hindi.csv", index=False)
    print("All Hindi responses saved to results/claude_responses_hindi.csv")

# Run the main function
if __name__ == "__main__":
    main()