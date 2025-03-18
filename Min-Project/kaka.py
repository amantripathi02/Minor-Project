from groq import Groq

client = Groq(api_key="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")

try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=1,
        max_tokens=100
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
