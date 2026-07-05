# NudiLLM: Machine Learning Interview Preparation Guide

---

## 1. Project Overview & Tech Stack
**Q: Tell me about your Kannada LLM project and the tech stack you used.**
* **Answer:** "I built a custom Large Language Model called NudiLLM, completely from scratch. It is a ~13.8 million parameter, Decoder-only Transformer model trained exclusively on Kannada text."
* **Tech Stack:**
  * **Core ML Framework:** PyTorch (for Tensor operations, Autograd, and Neural Network modules).
  * **Hardware Acceleration:** CUDA (NVIDIA GTX 1650, 4GB VRAM).
  * **Data Processing & Tokenization:** SentencePiece (for training the custom Byte-Pair Encoding tokenizer), NumPy (for efficient memory-mapped binary caching).
  * **Data Source:** HuggingFace datasets (AI4Bharat Kannada Wikipedia).

**Q: Did you just use a prebuilt model from HuggingFace?**
* **Answer:** "No, this was built 100% from scratch. I did not import `Llama`, `GPT2`, or use `from_pretrained()`. I wrote the actual neural network architecture—including the Multi-Head Self-Attention mechanisms and Transformer blocks—using base PyTorch modules (`nn.Linear`, `nn.Embedding`)."

---

## 2. System Design & Pipeline
**Q: Walk me through the system design of your LLM training pipeline.**
* **Answer:** "The pipeline is divided into four modular steps:
  1. **Data Ingestion:** I downloaded the Kannada Wikipedia corpus and cleaned the raw text, saving it to disk.
  2. **Tokenization:** I trained a SentencePiece BPE tokenizer on the text to create a custom 8,000-token Kannada vocabulary.
  3. **Binary Caching:** To avoid memory bottlenecks on my 4GB GPU, I pre-tokenized all 36 million tokens and saved them as binary `.npy` arrays using NumPy. This allowed the Dataloader to instantly load the data into RAM.
  4. **Training Loop:** I used a manual PyTorch training loop with `AdamW` optimization, gradient clipping, and cosine learning rate decay, logging checkpoints and loss values periodically."

---

## 3. The Transformer Architecture
**Q: Can you explain the specific Transformer architecture you built?**
* **Answer:** "I built a **Decoder-only Transformer** (similar to GPT-2). It consists of:
  * **Embedding Layer:** Maps the 8,000 Token IDs into dense mathematical vectors (dim=384).
  * **Positional Encoding:** Adds spatial awareness, so the model knows the order of the words.
  * **Transformer Blocks (6 Layers):** Inside each block, there are two main components:
    1. **Multi-Head Self-Attention (6 Heads):** This allows the model to look at the entire context of a sentence at once and weigh which words are the most relevant to each other.
    2. **Feed-Forward Network:** A standard multi-layer perceptron that processes the output of the attention layer.
  * **Language Modeling Head:** A final linear layer that outputs the raw probabilities (logits) across all 8,000 tokens in the vocabulary."

---

## 4. Data & Tokenization
**Q: Why a vocabulary size of exactly 8,000?**
* **Answer:** "It was an architectural decision based on hardware constraints. The vocabulary size directly impacts the size of the Embedding layer and the final Output layer. A massive vocabulary (like 50,000) would have exceeded the 4GB VRAM limit on my GTX 1650. 8,000 was the perfect sweet spot for efficient Kannada representation while fitting in memory."

---

## 5. The Core of LLMs (How it learns)
**Q: What is the core objective function of an LLM?**
* **Answer:** "Next-Token Prediction. We take a sequence of Kannada tokens, hide the last token, and ask the model to predict it. We use **Cross-Entropy Loss**, which measures the difference between the model's predicted probability distribution and the actual correct word. The optimizer then adjusts the weights to minimize that loss."

---

## 6. Training Mechanics: Epochs, Steps, and PyTorch
**Q: Explain what Epochs and Steps mean in your training loop.**
* **Answer:** 
  * A **Step** is a single forward/backward pass processing one batch of text.
  * An **Epoch** is one complete pass through the entire dataset (~17,750 steps for my 36M token dataset).

**Q: Why did you train for exactly 3 Epochs?**
* **Answer:** "To avoid overfitting (the 'parrot effect'). Following modern scaling laws (like Chinchilla), 3 epochs provided the optimal balance of learning Kannada grammar without simply memorizing Wikipedia word-for-word."

---

## 7. Inference & Debugging
**Q: During early testing, the model got stuck in a repetition loop. Why does this happen?**
* **Answer:** "This is known as 'Degeneration'. In early epochs, the model correctly learned the mathematical association between highly related words, but lacked the contextual depth to form a full sentence. It got stuck selecting the highest-probability related tokens in an infinite loop. This resolves naturally as training progresses."

---

## 8. Deployment
**Q: In traditional ML, we deploy a `.pkl` file. How do you deploy NudiLLM?**
* **Answer:** "LLM deployment requires three distinct components:
  1. **The Weights (`best.pt`):** The binary file containing the 13.8 million learned parameters.
  2. **The Architecture (`nudi.py`):** The PyTorch code that defines the skeleton of the neural network.
  3. **The Tokenizer (`kannada_bpe.model`):** The translator required to convert raw text into numerical Token IDs, and vice versa."
