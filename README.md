# Building a GPT Model from Scratch

A clean, from-the-ground-up implementation of the **GPT-2 (124M) architecture** in PyTorch.
The notebook builds the full decoder-only Transformer one component at a time — read it
top-to-bottom and you have a complete, working GPT.

## What's inside

[`GPT_Model_From_Scratch.ipynb`](GPT_Model_From_Scratch.ipynb) walks through every piece,
each defined exactly once, in the order it's used:

1. Model configuration (GPT-2 124M hyper-parameters)
2. Imports & BPE tokenization (`tiktoken`)
3. Layer Normalization
4. GELU activation (tanh approximation)
5. Position-wise Feed-Forward network
6. Multi-head **causal** self-attention
7. Pre-LayerNorm Transformer block with residual connections
8. The full `GPTModel`
9. Parameter count & memory footprint
10. Greedy autoregressive text generation

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

## Run it

Open the notebook in Jupyter / VS Code and run all cells, or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace GPT_Model_From_Scratch.ipynb
```

## Expected output

The model assembles to the correct size:

```
Total parameters:            163,009,536
Parameters with weight tying: 124,412,160
Approx. model size:           621.83 MB
```

The final cell runs the full forward + generation pipeline. The weights are **random
(untrained)**, so the generated text is intentionally gibberish — it proves the pipeline
works end to end:

```
Decoded text: Hello, I am Featureiman Byeswickattribute argue
```

## Next steps

Train the model on real text so the generated output becomes coherent.

## Credits

The architecture follows the design in Sebastian Raschka's *Build a Large Language Model
(From Scratch)*.
