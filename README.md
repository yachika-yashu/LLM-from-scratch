# Building a GPT Model from Scratch

A from-the-ground-up implementation of the **GPT-2 architecture** in PyTorch, taken all the
way from an empty file to a model that follows instructions.

Nothing is imported from a transformer library. Every component — attention, layer norm,
the transformer block, the training loop, the decoding strategies — is written out and
explained, and each notebook states *why* a design choice was made, not just what it does.

## The path through the repository

Read the notebooks in this order. Each one picks up where the previous one ended.

### 1. [`GPT_Model_From_Scratch.ipynb`](GPT_Model_From_Scratch.ipynb) — build and train

The full decoder-only transformer, one component at a time, each defined exactly once in
the order it is used:

1. Model configuration (GPT-2 124M hyper-parameters)
2. Imports and BPE tokenization (`tiktoken`)
3. Layer normalization
4. GELU activation (tanh approximation)
5. Position-wise feed-forward network
6. Multi-head **causal** self-attention
7. Pre-LayerNorm transformer block with residual connections
8. The full `GPTModel`
9. Parameter count and memory footprint
10. Greedy autoregressive text generation
11. Cross-entropy loss
12. Perplexity
13. The training corpus (`the-verdict.txt`)
14. Batching: `Dataset` and `DataLoader`
15. Loss over a data loader
16. The training loop
17. Decoding strategies: temperature and top-k
18. Saving and loading weights

The model assembles to the correct size:

```
Total parameters:            163,009,536
Parameters with weight tying: 124,412,160
Approx. model size:           621.83 MB
```

Trained on a single short story, the output is grammatical but narrow — which is exactly
why the next notebook exists.

### 2. [`loading_weights.ipynb`](loading_weights.ipynb) — load OpenAI's pretrained weights

Downloads the original GPT-2 checkpoint and copies it into our `GPTModel`. This is where
the model stops producing gibberish. The interesting part is that the copy is not
mechanical — three mismatches have to be reconciled:

- OpenAI's names differ from ours (`wte` versus `tok_emb`)
- their weight matrices are stored transposed, because GPT-2 used a `Conv1D` layer
- query, key and value are fused into one `[768, 2304]` matrix and must be split

### 3. [`Classification fine tuning/finetuning.ipynb`](Classification%20fine%20tuning/finetuning.ipynb) — classification finetuning

Turns the pretrained model into a spam classifier on the SMS Spam Collection dataset:
the 50,257-way output head is replaced with a 2-way head, most of the network is frozen,
and loss is computed from the last token only.

```
Training accuracy:   97.21%
Validation accuracy: 97.32%
Test accuracy:       95.67%
```

The trade-off is stated explicitly: the model can now emit only its two class labels and
is no longer a general-purpose text generator.

### 4. [`Instruction Fine tuning/instruction_finetuning.ipynb`](Instruction%20Fine%20tuning/instruction_finetuning.ipynb) — instruction finetuning

The opposite approach. The output head is left alone and the whole model (gpt2-medium,
355M) is trained on 1,100 instruction-response pairs in Alpaca format. Covers the parts
that are easy to get wrong:

- a collate function that pads each batch to *its own* longest sequence, not the
  dataset's, worked through by hand on a three-example batch
- padding positions masked with `-100` so cross entropy ignores them, while one
  end-of-text token is kept so the model learns when to stop
- automated evaluation by scoring responses with a larger local model (Llama 3 via
  Ollama), since there is no single correct answer to compare against

This notebook is self-contained: the weight-copying utilities from notebook 2 and the
generation utilities from notebook 1 are repeated inline, each one placed immediately
before the step that needs it.

### Supporting files

| File | Purpose |
| --- | --- |
| [`gptmodel.py`](gptmodel.py) | The `GPTModel` class extracted from notebook 1, imported by the rest |
| [`gpt_download3.py`](gpt_download3.py) | Downloads OpenAI's checkpoint and reshapes the flat TF variable names into a nested dict |
| [`the-verdict.txt`](the-verdict.txt) | The short-story training corpus for notebook 1 |
| [`images/`](images/) | Diagrams used in the notebooks |

Notebooks 3 and 4 live in their own folders and each keeps a copy of the modules it
imports, so a folder can be opened and run on its own without the repository root on the
path.

## Setup

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

TensorFlow is in the requirements only to read OpenAI's original checkpoint format; no
part of the model uses it.

Notebook 4 additionally needs [Ollama](https://ollama.com) running locally with the
`llama3` model pulled, for the evaluation step.

## Running

Open a notebook in Jupyter or VS Code and run all cells, or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace GPT_Model_From_Scratch.ipynb
```

Everything large that is downloaded or produced along the way — the GPT-2 checkpoint, the
SMS Spam Collection dataset and its train/validation/test splits, and the saved `.pth`
weights — is gitignored, and all of it is regenerated by running the notebooks in order.
The 204 KB instruction dataset and the model's responses to the test set are small enough
to be committed, so notebook 4 can be read without downloading anything.

Training times on the machine these notebooks were run on: about 15 minutes for the spam
classifier and about 2 hours for instruction finetuning.

## Credits

The architecture and the progression of chapters follow Sebastian Raschka's
*Build a Large Language Model (From Scratch)*.
