# **Adaptive Prompt Engineering for Gender Bias Mitigation in Multilingual Large Language Models**  

## **Overview**  
This research investigates gender bias in Large Language Models (LLMs) by analyzing responses from three different models: **OpenAI’s GPT, Google Gemini, and Microsoft Copilot**. By collecting responses to identical prompts and applying computational bias detection techniques, this study aims to identify and mitigate gender bias through **prompt engineering**.  

## **Data Collection**  
Responses from each LLM have been saved into three separate CSV files:  
- **`prompts_openai.csv`** – Responses from GPT  
- **`prompts_gemini.csv`** – Responses from Gemini  
- **`prompts_copilot.csv`** – Responses from Copilot  

Each dataset consists of prompts designed to explore potential gender bias across various domains such as **education, healthcare, leadership, recruitment and social**.  

## **Bias Detection Methods**  
To quantify gender bias, the study employs three computational techniques:  

1. **Word Embedding Association Test (WEAT)**  
   - Measures associations between gendered terms and other word categories.  
   - Helps identify implicit biases in language patterns.  

2. **Sentiment Analysis**  
   - Assesses whether responses containing gendered terms exhibit systematic emotional tone variations.  
   - Detects potential preference for or against certain gender identities.  

3. **Lexical Diversity & Frequency Analysis**  
   - Examines whether male- or female-associated words appear disproportionately in generated text.  
   - Identifies language patterns that may indicate embedded bias.  

These methods allow for the detection of both overt and subtle biases within model-generated responses.  

## **Bias Mitigation Through Prompt Engineering**  
Following initial bias detection, prompt engineering techniques will be applied to reduce gender bias and the process will be repeated.  

### **Proposed Prompt Engineering Strategies:**  
1. **Counterfactual Prompting**  
   - Swaps gendered terms to compare responses and detect discrepancies in model behavior.  

2. **Debiasing Instructions**  
   - Explicitly instructs models to generate neutral or fair responses.  

3. **Re-ranking Techniques**  
   - Generates multiple responses and selects the least biased output for analysis.  

## **Next Steps**  
- Implement prompt engineering and re-run bias detection.  
- Compare results before and after mitigation techniques.  
- Conduct qualitative analysis to evaluate the effectiveness of different strategies.  

## **Conclusion**  
This study aims to develop a structured methodology for **detecting and mitigating gender bias in LLMs** through **systematic prompt engineering**. The findings will contribute to improving fairness in AI-generated text and guiding ethical AI development.  