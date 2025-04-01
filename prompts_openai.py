import openai

openai.api_key = "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"

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