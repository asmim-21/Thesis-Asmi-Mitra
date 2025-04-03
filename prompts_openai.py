from openai import OpenAI
import pandas
import os

OPENAI_API_KEY = "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"

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY
)

response = client.responses.create(
    model="gpt-4o",
    input="Describe the strengths of an ideal student in English and History",
)

print(response.output_text)