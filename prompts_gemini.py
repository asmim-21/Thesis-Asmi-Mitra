from google import genai
import pandas as pd

# Set up Google Gemini API client
API_KEY = "AIzaSyBXOPG5i8JLEidQKv3uyao9ijZS0OfO4ZA"
client = genai.Client(api_key=API_KEY)

# List of prompts to test
prompts = [
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

prompts_length = len(prompts)

# Collect responses
results = []

print(f"Processing {prompts_length} prompts\n")

for i, prompt in enumerate(prompts, start=1):
    print(f"➡️  {i}/{prompts_length}: Sending prompt...")

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    
    results.append({"Prompt": prompt, "Response": response.text})
    
    print(f"✔️  {i}/{prompts_length}: Done\n")

# Convert to DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("results/gemini_responses.csv", index=False)

print("✅  All responses saved to gemini_responses.csv")