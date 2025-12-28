from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./gpt2-finetuned",
    tokenizer="./gpt2-finetuned",
    device=-1  # CPU
)

prompt = "Generative artificial intelligence is"

result = generator(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
    truncation=True,
    pad_token_id=50256
)

print("Prompt:")
print(prompt)
print("\nGenerated Text:\n")
print(result[0]["generated_text"])
