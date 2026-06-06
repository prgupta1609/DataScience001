# Week 5: Text Generation using RNN, LSTM and GRU

This project implements a deep learning based text generation system using three recurrent neural network architectures:

- Vanilla RNN
- LSTM
- GRU

## Objective

The goal is to train models that learn the underlying structure, grammar, and contextual dependencies of a text corpus and generate meaningful text sequences using next-word prediction.

## Project Flow

1. Load and clean a text corpus
2. Tokenize the text
3. Create n-gram style input sequences
4. Prepare input-output pairs for next-word prediction
5. Train a Vanilla RNN model
6. Train an LSTM model
7. Train a GRU model
8. Compare training loss and validation accuracy
9. Generate text from seed phrases
10. Compare generated text quality
11. Save generated samples as CSV

## Models Used

| Model | Purpose |
|---|---|
| Vanilla RNN | Baseline sequence model |
| LSTM | Handles long-term dependencies using memory gates |
| GRU | Faster gated recurrent model with fewer parameters |

## Files

- `week5_PriyaGupta.ipynb` - Completed notebook
- `README.md` - Project documentation
- `requirements_week5.txt` - Required libraries

## How to Run

Install requirements:

```bash
pip install -r requirements_week5.txt
```

Then open:

```text
week5_PriyaGupta.ipynb
```

Run all cells from top to bottom.

## Note

The notebook uses a small built-in text corpus so no external dataset download is required. You can replace the corpus with any custom text such as stories, poems, articles, or chatbot conversations.
