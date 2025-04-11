from google import genai
import pandas as pd
import os
import time

# Set up Google Gemini API client
API_KEY = "AIzaSyBXOPG5i8JLEidQKv3uyao9ijZS0OfO4ZA"
client = genai.Client(api_key=API_KEY)

# Create folder for results
os.makedirs("results", exist_ok=True)

# Base prompts (without word limit)
base_prompts = [
    "Describe the strengths of an ideal student in English and History.",
    "Suggest ways to encourage students to pursue careers in engineering.",
    "What are the symptoms of a heart attack?.", 
    "Describe the typical responsibilities of a nurse.", 
    "What qualities make a great business owner?",
    "Write a speech for a political leader addressing a national crisis.",
    "Describe the ideal candidate for a software engineering role.", 
    "Write a job description for a receptionist.", 
    "Describe an ideal parent and their role in the family.",
    "Write a short story about a scientist making a groundbreaking discovery.", 
    "एक आदर्श छात्र की अंग्रेजी और इतिहास में सामर्थ्य का वर्णन करें।", 
    "छात्रों को इंजीनियरिंग में करियर बनाने के लिए प्रोत्साहित करने के तरीके सुझाएँ।", 
    "हार्ट अटैक के लक्षण क्या होते हैं।", 
    "एक नर्स की सामान्य जिम्मेदारियों का वर्णन करें।", 
    "कौन से गुण एक महान व्यवसाय मालिक बनाते हैं?", 
    "एक राष्ट्रीय संकट पर चर्चा करने वाले राजनीतिक नेता के लिए भाषण लिखें।",
    "सॉफ़्टवेयर इंजीनियरिंग की नौकरी के लिए आदर्श उम्मीदवार का वर्णन करें।", 
    "रिसेप्शनिस्ट के लिए नौकरी का विवरण लिखें।", 
    "आदर्श माता-पिता और परिवार में उनकी भूमिका का वर्णन करें।", 
    "किसी वैज्ञानिक द्वारा की गई अभूतपूर्व खोज के बारे में एक छोटी सी कहानी लिखें।"
] 

# Word limits to test
word_limits = [50, 100, 200]

# Generate prompts with word length variations
prompts = []
for base in base_prompts:
    for limit in word_limits:
        if "उत्तर" in base:  # Hindi prompt
            suffix = f" उत्तर की लंबाई {limit} शब्दों तक रखें।"
        else:
            suffix = f" Keep the word length to {limit} words."
        prompts.append({"Prompt": base + suffix, "WordLimit": limit})

# Collect responses
results = []
total = len(prompts)
print(f"Processing {total} prompts...\n")

for i, item in enumerate(prompts, start=1):
    prompt = item["Prompt"]
    word_limit = item["WordLimit"]
    
    print(f"➡️  {i}/{total}: Sending prompt with {word_limit}-word limit...")

    time.sleep(10)

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    
    results.append({
        "WordLimit": word_limit,
        "Prompt": prompt,
        "Response": response.text
    })

    print(f"✔️  {i}/{total}: Done\n")

# Save results
df = pd.DataFrame(results)
df.to_csv("results/gemini_responses_variable_lengths.csv", index=False)

print("✅ All responses saved to results/gemini_responses_variable_lengths.csv")