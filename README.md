# **Adaptive Prompt Engineering for Gender Bias Mitigation in Multilingual Large Language Models**  

**Written By: Asmi Mitra**  
Supervisor: Dr. Shahadat Uddin    
A thesis submitted in fulfilment of the requirements for the degree of Bachelor of Engineering Honours (Software)  

School of Electrical & Computer Engineering   
Faculty of Engineering   
The University of Sydney  

## **Overview**  
This research investigates gender bias in Large Language Models (LLMs) by analyzing responses from three different models: **OpenAI’s GPT, Google Gemini, and Anthropic Claude**. By collecting responses to identical prompts and applying computational bias detection techniques, this study aims to identify and mitigate gender bias through **prompt engineering** in English and Hindi.  

## **Data Collection**  
Each LLM has its own results folder and associated Python scripts for processing and analysis.
```
├── figures/
│   ├── claude/
│   │   ├── bias_detection/
│   │   └── prompt_engineering/
│   ├── ...
│   └── summary/
│       ├── heatmap_model_language_weat.png
│       ...
├── results/
│   ├── claude/
│   │   ├── bias_detection/
│   │   ├── initial_responses/
│   │   ├── prompt_engineering/
│   ├── ...
│   └── extracted_questions/
│       ├── claude_base_english_selected_questions.csv
│       ├── claude_base_hindi_selected_questions.csv
│       ├── claude_explicit_instruction_english_selected_questions.csv
│       ...
├── scripts/
│   ├── claude/
│   ├── gemini/
│   ├── openai/
│   └── ... (python scripts for response collection, evaluation and prompt engineering)
```  

Each dataset consists of LLM responses to prompts designed to explore potential gender bias across various domains such as **education, healthcare, leadership, recruitment and social**. Responses are collected in three different lengths: 50, 100 and 200 words, and in both English and Hindi.   

## **Bias Detection Methods**  
To quantify gender bias, the study employs three computational techniques:  

1. **Word Embedding Association Test (WEAT)**  
   - Measures associations between gendered terms and other word categories.  
   - Helps identify implicit biases in language patterns.  

2. **Sentiment Analysis**  
   - Assesses whether responses containing gendered terms exhibit systematic emotional tone variations.  
   - Detects potential preference for or against certain gender identities.  

3. **Lexical Diversity & Frequency Analysis**  
   - Examines whether male or female associated words appear disproportionately in generated text.  
   - Identifies language patterns that may indicate embedded bias.  

These methods allow for the detection of both overt and subtle biases within model-generated responses.  

## **Bias Mitigation Through Prompt Engineering**  
Following initial bias detection, prompt engineering techniques are applied to reduce gender bias and the bias detection process is repeated to measure improvements.  

### **Prompt Engineering Strategies:**  
1. **Explicit Instruction Prompting**
   
   Prompts include direct instructions asking the model to generate unbiased, fair, or gender-neutral responses.
   
   Example: 

      ```Please ensure the response is gender-neutral and unbiased.```

   This method aims to guide the model’s output behavior clearly by foregrounding fairness in the prompt itself.

2. **Few-Shot Prompting**
   
   Prompts are accompanied by a few example question-response pairs that model ideal, unbiased behaviour. By showing how to respond in a neutral way, the model is encouraged to follow the implicit structure and tone of the examples.
   
   Benefits:
   - Helps the model generalize unbiased behaviour across unseen prompts.
   - Useful in both zero-resource and multilingual contexts.

3. **Role-Based Prompting**
   
   Prompts place the model in a hypothetical role (e.g., human rights officer, unbiased hiring manager, etc.) to frame responses through a professional or ethical lens.
   
   Example:
   
   ```You are an unbiased language consultant who writes in a gender-neutral and inclusive manner.```
   
   This technique leverages the model’s contextual understanding of roles to encourage fairer, more socially responsible outputs.

## **Conclusion**  
This study aims to develop an innovative and structured methodology for **detecting and mitigating gender bias in multilingual LLMs** through **systematic prompt engineering**. The findings aim to contribute to improving fairness in AI-generated text and guiding ethical AI development.  
