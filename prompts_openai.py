import openai

openai.api_key = "your_api_key"

def get_ai_response(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Example Usage:
prompt = "Describe an ideal leader."
response = get_ai_response(prompt)
print(response)