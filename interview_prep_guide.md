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

---

## 9. NudiLLM Architecture Diagram

**Q: Can you sketch out the high-level architecture of NudiLLM?**
* **Answer:** "NudiLLM is a standard Auto-Regressive Decoder-Only Transformer (similar to GPT-2). Here is the flow of data through the network:"

```text
 ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
 │     Input Stage      │ ───> │      Embeddings      │ ───> │  Transformer Blocks  │
 │                      │      │                      │      │      (x6 Layers)     │
 │                      │      │                      │      │                      │
 │ • Input Text         │      │ • Token Embedding    │      │ • Layer Norm 1 & 2   │
 │ • BPE Tokenizer      │      │ • Positional Embed   │      │ • Multi-Head Attn    │
 │ • Token IDs          │      │ • Addition (+)       │      │ • Feed-Forward (GELU)│
 └──────────────────────┘      └──────────────────────┘      └──────────────────────┘
                                                                        │
                                                                        v
                               ┌──────────────────────┐      ┌──────────────────────┐
                               │        Output        │ <─── │     Output Layer     │
                               │                      │      │                      │
                               │ • Probabilities      │      │ • Final Layer Norm   │
                               │ • Next Word/Token    │      │ • Linear Output Head │
                               │                      │      │ • Softmax            │
                               └──────────────────────┘      └──────────────────────┘
```

---

## 10. Hardware vs Architecture
**Q: Why do we need a GPU instead of a CPU?**
* **Answer:** "A CPU is like a professor—it can solve complex logic sequentially but only has a few cores. A GPU is like a stadium of 5th graders—it has thousands of cores (my GTX 1650 has 896 CUDA cores) that can perform simple matrix multiplications simultaneously. This Parallel Processing reduces training time from years to hours."

**Q: How did you fit a 13.8M parameter model onto a 4GB GPU?**
* **Answer:** "I used **FP16 Mixed Precision**. Standard PyTorch uses 32-bit floats. By casting operations to 16-bit (using `torch.autocast`), I cut the memory footprint of the weights and activations in half, allowing the model to train using only ~2.5GB of VRAM."

---

## 11. Core Mathematical Concepts
**Q: What exactly is a 'Parameter' in your model?**
* **Answer:** "A parameter is an adjustable number (a weight or bias) in the model's massive mathematical equation. My model has 13,797,120 of these numbers. During training, the optimizer continuously adjusts these decimal values until the equation correctly outputs Kannada text."

**Q: Explain how Token Embeddings and Positional Embeddings work.**
* **Answer:** "A Token Embedding stretches a single integer (Token ID) into a dense 384-dimensional array, placing the word as a coordinate in a 'galaxy of meaning' (e.g., King and Queen are close together). Because Transformers read all words in parallel, they have no concept of time or order. We physically add a Positional Embedding (another 384-dim array representing spatial order) so the AI knows both *what* the word means and *where* it is in the sentence."

**Q: Why did you use GELU instead of ReLU for the Feed-Forward Network?**
* **Answer:** "ReLU has a harsh, robotic cutoff (instantly dropping negative numbers to zero). GELU uses a smooth, probabilistic Gaussian curve. Because human language is full of nuance and probabilities, GELU allows for smoother gradient flow during backpropagation, making it the industry standard for modern LLMs."

**Q: Why AdamW instead of standard SGD or Adam?**
* **Answer:** "Adam provides adaptive learning rates for every single parameter, making training fast. The 'W' stands for Weight Decay. Weight Decay mathematically shrinks the parameters toward zero on every step. This forces the network to generalize and learn the language rather than lazily making a few weights massive to memorize the training data (overfitting)."

---

## 12. Inference (Generation)
**Q: How does the model generate a full paragraph of text?**
* **Answer:** "It uses **Autoregressive Generation**. It predicts one single word, glues that word to the original prompt, and feeds the entire new sentence back into itself in a loop until it outputs the `<eos>` (End of Sentence) token."

**Q: What sampling parameters do you use to stop it from sounding robotic?**
* **Answer:** "I use **Temperature** (e.g., 0.8) to inject creativity by softening the probabilities. I use **Top-K** (e.g., 50) and **Top-P** to filter out gibberish words, forcing the model to only pick from the most highly probable subset. Finally, I use a **Repetition Penalty** (1.3) to mathematically punish the model for reusing words, preventing infinite loops."

---

## 13. Advanced Tokenization & Data Prep
**Q: What is the difference between the `.model` file and the `.npy` file?**
* **Answer:** "The `kannada_bpe.model` is the dictionary rulebook (mapping words to numbers). The `train.tokens.npy` file is the actual 393MB text corpus completely translated into numbers using that rulebook. The GPU only ever reads the `.npy` file during training."

**Q: What are the 3 files generated by the Tokenizer?**
* **Answer:** 
  1. `kannada_bpe.model`: The actual binary translation engine used by the code.
  2. `kannada_bpe.vocab`: A human-readable text file showing all 8,000 learned tokens for debugging.
  3. `tokenizer_config.json`: Metadata instructing the code on special tokens (like `<eos>`, `<unk>`).

**Q: If your vocabulary is only 8,000 words, what happens if a user types a rare word? Does it skip it?**
* **Answer:** "No, it never skips. BPE is a *sub-word* tokenizer. Its base vocabulary contains every individual alphabet character. If it sees a rare word like 'ಕ್ರಿಪ್ಟೋಕರೆನ್ಸಿ', it simply falls back and breaks it into smaller known chunks or individual letters. It can mathematically tokenize any word in the universe."

**Q: What is the difference between NFC Normalization and Tokenization?**
* **Answer:** "NFC is a data-cleaning step that happens *first*. It ensures individual characters (like 'ಕೆ') are stored safely as one block on the hard drive. Tokenization happens *second*, where it takes those safe characters and groups them into words. If I skipped NFC, the Tokenizer would rip the characters apart."

---

## 14. Advanced Architecture (Encoder vs Decoder)
**Q: Why did you use a Decoder-only architecture instead of an Encoder?**
* **Answer:** "Encoders (like BERT) read entire documents at once for classification/analysis. Decoders are built to generate text one word at a time. Since NudiLLM is a chatbot, a Decoder-only architecture with Causal Masking forces it to learn 'next-token prediction', which is perfect for generation."

**Q: Since ChatGPT is Decoder-only, how does it summarize articles? (Summarization usually requires an Encoder)**
* **Answer:** "It uses a trick called **In-Context Learning**. You paste the article into the prompt. The Decoder reads it left-to-right, builds a massive mathematical understanding of the article in its attention layers, and when it reaches the instruction 'Summarize this', the most probable 'next words' naturally form a perfect summary. It turns a Sequence-to-Sequence task into a Next-Word Prediction task."

**Q: How has the LLM architecture landscape evolved in 2026?**
* **Answer:** "While the core Decoder-only Transformer is still the foundation, three major shifts happened: 1) **Mixture of Experts (MoE)** to run massive models cheaply by routing tasks to sub-networks, 2) **Native Multimodality** (processing audio/video directly in the Decoder without external translators), and 3) **SSM/Mamba Hybrids** to allow infinite context windows without crashing GPU memory."

---

## 15. The Lifecycle of a Token
**Q: Walk me step-by-step through exactly what happens to a token inside your Transformer.**
* **Answer:** 
  1. **Embeddings:** The token ID is stretched into a 384-dim array (meaning), and positional data (time) is added.
  2. **LayerNorm 1:** The numbers are statistically standardized to prevent math explosions.
  3. **Causal Self-Attention:** The word creates a Query, looks for matching Keys in past words, and multiplies by Values to build context. The Causal Mask blocks it from seeing the future.
  4. **Residual Connection:** The original word is added back in to prevent data loss.
  5. **LayerNorm 2 & FFN:** The context is standardized and passed through a Feed-Forward Network (using GELU) to process and 'think' about the new context.
  6. **Output Head:** After passing through 6 layers, the final array hits the Language Model Head, which projects it back into 8,000 probabilities to select the next word.

---

## 16. Hard Numbers Cheat Sheet
* **Total Parameters:** 13.8 Million
* **Transformer Layers (Blocks):** 6
* **Attention Heads:** 6
* **Embedding Dimension:** 384
* **Vocabulary Size:** 8,000 tokens (BPE)
* **Context Window (Sequence Length):** 256 tokens
* **Batch Size:** 8
* **Precision:** FP16
* **VRAM Usage:** ~2.5 GB (on a 4GB GTX 1650)
* **Dataset Size:** 393 MB raw text (Kannada Wikipedia)
