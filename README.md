

Text Generation with GPT-2 (Fine-Tuned on Generative AI Content)
📌 Project Overview

This project demonstrates text generation using a fine-tuned GPT-2 model, a transformer-based language model. The objective is to train the model on a custom Generative AI–focused dataset so that it can generate coherent, contextually relevant text based on a given prompt.

Instead of training a language model from scratch, a pre-trained GPT-2 model is fine-tuned using domain-specific text. This approach significantly reduces computational cost while maintaining high-quality language generation.

🎯 Objectives

To understand the working of transformer-based language models

To fine-tune GPT-2 on a custom text dataset

To generate meaningful text from a user-defined prompt

To demonstrate practical usage of Natural Language Processing (NLP) concepts

🧠 Technologies Used

Python

PyTorch

Hugging Face Transformers

GPT-2 (Causal Language Model)

Natural Language Processing

🏗️ Model Architecture

GPT-2 is based on the Transformer Decoder architecture, which relies on:

Self-attention mechanism

Positional encoding

Multi-head attention

Feed-forward neural networks

The model is trained using causal language modeling, where it learns to predict the next token in a sequence based on previous tokens.

📂 Project Structure
language-modeling/
│
├── data/
│   └── train.txt              # Custom training dataset
│
├── gpt2-finetuned/            # Fine-tuned model & tokenizer
│
├── run_clm.py                 # Training script
├── generate.py                # Text generation script
├── requirements.txt
└── README.md

📊 Dataset Description

Type: Plain text (.txt)

Domain: Generative Artificial Intelligence & NLP

Content: Academic and technical descriptions related to AI, transformers, and language models

Purpose: To adapt GPT-2 to generate domain-specific text

⚙️ Model Training

The GPT-2 model was fine-tuned using the Hugging Face run_clm.py script.

Training Command
python run_clm.py \
  --model_name_or_path gpt2 \
  --train_file data/train.txt \
  --do_train \
  --output_dir ./gpt2-finetuned \
  --per_device_train_batch_size 2 \
  --num_train_epochs 3 \
  --block_size 128 \
  --overwrite_output_dir

Training Details

Epochs: 3

Batch Size: 2

Block Size: 128 (used to handle a smaller dataset efficiently)

Device: CPU

Final Training Loss: ~3.5

✨ Text Generation

After fine-tuning, the model is used to generate text from a given prompt.

Generation Script (generate.py)
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./gpt2-finetuned",
    tokenizer="./gpt2-finetuned",
    device=-1
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

Sample Output
Prompt:
Generative artificial intelligence is

Generated Text:
Generative artificial intelligence is used to generate complex neural networks
and improve natural language processing systems by learning contextual patterns.

📈 Results and Observations

The fine-tuned model generates context-aware and coherent text

Output reflects the style and domain of the training dataset

Demonstrates the effectiveness of transfer learning using GPT-2

⚠️ Limitations

Small dataset limits long-range coherence

CPU-based training restricts scalability

GPT-2 may still generate generic or repetitive text for longer outputs

🚀 Future Enhancements

Increase dataset size for better generalization

Train on GPU for improved performance

Experiment with advanced sampling techniques

Deploy as a web application using Streamlit or Flask

📚 Conclusion

This project successfully demonstrates fine-tuning GPT-2 for domain-specific text generation. By leveraging pre-trained transformer models, high-quality language generation can be achieved efficiently, making this approach suitable for academic and real-world NLP applications.

👤 Author

Mohammed Muneeb
Project: Text Generation with GPT-2
