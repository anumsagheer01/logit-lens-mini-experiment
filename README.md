<p align="center">
  <img src="./assets/anthropic-header.svg" alt="Looking Inside Language Models" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-D97757?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TransformerLens-Interpretability-F7F4ED?style=for-the-badge&labelColor=191919" alt="TransformerLens">
  <img src="https://img.shields.io/badge/Research-Logit%20Lens-191919?style=for-the-badge" alt="Logit Lens">
</p>

<p align="center">
  <b>Can a language model represent a concept internally even when that concept never appears in its output?</b>
</p>

---

## Overview

This is a small mechanistic interpretability experiment that looks at how next-token predictions change across transformer layers.

It was inspired by Anthropic's paper *Verbalizable Representations Form a Global Workspace in Language Models*, which studies whether language models contain internal representations that can be used in reasoning and later expressed in words. The paper introduces the **Jacobian Lens** (J-lens) to study these representations, and calls the relevant internal region **J-space**.

This notebook does not reproduce the Jacobian Lens method. It uses the simpler **logit lens** technique to inspect the model's residual stream after each layer.

**Contents**

- [The Experiment](#the-experiment)
- [Models Tested](#models-tested)
- [How the Logit Lens Works](#how-the-logit-lens-works)
- [Logit Lens vs Jacobian Lens](#logit-lens-vs-jacobian-lens)
- [Results](#results)
- [Main Finding](#main-finding)
- [Connection to Anthropic's Paper](#connection-to-anthropics-paper)
- [Limitations](#limitations)
- [Possible Next Steps](#possible-next-steps)
- [Installation](#installation)
- [Key Terms](#key-terms)

---

## The Experiment

Each model received this prompt:

```text
Concentrate on the concept of oranges.
Now copy this sentence exactly:
The old painting hung crookedly on the
```

The prompt introduces two competing ideas:

1. A hidden concept to concentrate on: **oranges**
2. A sentence that should end with: **wall**

The experiment checks whether orange-related tokens show up in the model's intermediate predictions, even though the sentence it's asked to produce has nothing to do with oranges.

---

## Models Tested

| Model | Approximate size | Model type |
| :--- | :--- | :--- |
| GPT-2 | 124M parameters | Base language model |
| Qwen1.5-0.5B-Chat | 500M parameters | Instruction-tuned chat model |
| Qwen1.5-1.8B-Chat | 1.8B parameters | Instruction-tuned chat model |

All models were loaded through TransformerLens.

---

## How the Logit Lens Works

A transformer stores and updates information in a shared internal vector called the **residual stream**. The notebook takes the residual stream after each layer and passes it through the model's final normalization and unembedding matrix.

```text
Intermediate residual stream
            |
            v
   Final layer normalization
            |
            v
     Unembedding matrix
            |
            v
   Vocabulary token scores
```

This gives a rough view of which tokens each layer appears to support. The main operation is:

```python
resid = cache["resid_post", layer][0, last_pos]
resid_normed = model.ln_final(resid)
layer_logits = resid_normed @ model.W_U
```

The notebook then records the five highest-scoring tokens at every layer.

---

## Logit Lens vs Jacobian Lens

**Logit lens.** Projects an intermediate activation straight into vocabulary space using the model's final output projection. It's simple, and it's useful for watching next-token predictions develop across layers. The catch is that early and late layers may use different internal coordinate systems, so early-layer readouts can look noisy or misleading.

**Jacobian lens.** Anthropic's method estimates how an earlier activation affects current and future outputs *after* passing through the remaining layers, which accounts for how representations shift with depth.

The two are related but not equivalent. Treat this project as a smaller experiment inspired by the paper's questions, not a reproduction of its method.

---

## Results

### GPT-2

GPT-2 showed the clearest progression across layers.

**Early layers** produced broad or uncertain tokens:

```text
same, latter, simplest, smallest
```

**Middle layers** started moving toward physical surfaces and locations:

```text
edges, horizon, shoulders, walls, floor
```

**Later layers** strongly supported the expected sentence ending:

```text
wall, walls, floor, porch, fireplace, ceiling
```

By the final layer, `wall` was the strongest prediction.

### Qwen1.5-0.5B-Chat

The smaller Qwen model produced noisy, multilingual tokens across many early and middle layers. Near the final layers, its predictions became more connected to the copying task:

```text
The, the, wall, old
```

Its hidden state got easier to interpret as it moved closer to the output.

### Qwen1.5-1.8B-Chat

The larger Qwen model also produced noisy early-layer predictions. From roughly layer 16 onward, predictions stabilized around the start of the copied sentence:

```text
The, the, old
```

Unlike GPT-2, the final token position was heavily influenced by the chat template and generation formatting.

---

## Main Finding

**No clearly orange-related token appeared in the top five predictions for any of the three models.** The models instead became increasingly focused on the sentence they were asked to produce.

This does not prove the models ignored the concept of oranges. It only shows that orange-related information was not visible:

- through this specific logit lens readout
- at the final prompt position
- among only the top five vocabulary predictions
- for this single prompt

The concept may still have been represented in another form, at another position, or below the top five rankings.

---

## Connection to Anthropic's Paper

The paper studies **verbalizable representations**: internal representations a model may be able to express in words or use during reasoning. It proposes that some of these form a sparse internal region called **J-space**, and connects that idea to a **global workspace**, where selected information becomes available across different parts of a system.

Key terms from the paper:

| Term | Meaning in the paper |
| :--- | :--- |
| Verbalizable representation | An internal concept that may be available for the model to express in language |
| J-space | A sparse region containing verbalizable representations |
| Global workspace | A shared space where selected information becomes available for flexible use |
| Jacobian Lens | A method for estimating how internal activations influence current and future outputs |
| Directed attention | A model's ability to focus on a concept based on an instruction |
| Causal intervention | Changing an internal activation to test whether it affects behavior |

This notebook explores a related question in a much simpler way: does a directed concept such as *oranges* become visible in intermediate token predictions?

### What the experiment actually shows

The clearest result isn't hidden orange information. It's the gradual formation of the final prediction. For GPT-2, the progression looked roughly like this:

```text
Broad token possibilities
          |
          v
Physical surfaces and locations
          |
          v
Sentence-relevant objects
          |
          v
        wall
```

The Qwen results show a real limitation of the basic logit lens: earlier layers can produce fragmented, multilingual, or unrelated tokens because those activations aren't yet aligned with the final output space.

---

## Limitations

- **This is not a Jacobian Lens implementation.** The notebook applies the final normalization and unembedding matrix directly to intermediate activations. Anthropic's method accounts for downstream transformations using Jacobians.
- **Only the top five tokens were inspected.** An orange-related token may have appeared at a lower rank.
- **Only one prompt was tested.** A single prompt isn't enough to establish a pattern.
- **Only the final prompt position was analyzed.** The orange concept may be stronger near where it was introduced.
- **Tokenization may hide related concepts.** Models split words like `orange`, ` oranges`, `fruit`, and `citrus` into different pieces.
- **Chat templates affect the comparison.** GPT-2 received the raw prompt, while the Qwen chat models received extra formatting tokens.
- **No causal test was performed.** The notebook reads activations but doesn't edit, remove, or inject them.

---

## Possible Next Steps

- Track the exact rank of `orange`, `oranges`, `fruit`, and `citrus` at every layer
- Inspect well beyond the top five tokens
- Compare the same sentence with and without the orange instruction
- Inspect every token position in the prompt, not just the last one
- Test many concepts and sentence types
- Compare base models against instruction-tuned models
- Plot target-token probability across layers
- Use a tuned lens or a Jacobian Lens
- Add activation steering or ablation experiments
- Test whether the hidden concept changes later generated text

---

## Project Structure

```text
.
├── logit_lens_experiment.ipynb
└── README.md
```

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install torch transformer-lens jupyter
```

### Run the notebook

```bash
jupyter notebook
```

Then open `logit_lens_experiment.ipynb` and run the cells in order.

Models are downloaded from Hugging Face on the first run. Qwen1.5-1.8B needs significantly more memory than GPT-2 and Qwen1.5-0.5B.

---

## Key Terms

| Term | Simple meaning |
| :--- | :--- |
| Mechanistic interpretability | Studying how a model's internal components produce its behavior |
| Activation | A numerical representation created inside the model |
| Residual stream | The shared internal vector that carries information through transformer layers |
| Logits | Raw scores for possible output tokens |
| Unembedding matrix | The matrix that converts hidden states into vocabulary scores |
| Logit lens | A method for reading intermediate activations as token predictions |
| Jacobian Lens | A method for estimating how intermediate activations affect later outputs |
| J-space | The paper's proposed region of verbalizable internal representations |
| Global workspace | A shared space where selected information can be used across processes |
| Activation steering | Changing internal activations to influence model behavior |
| Ablation | Removing or suppressing an internal component to test its role |

---

## Takeaway

This experiment didn't find a clear orange-related token in the top logit lens predictions. What it did show is how token predictions change across transformer layers, and how later layers become increasingly aligned with the final output.

It also shows why the distinction between the logit lens and Anthropic's Jacobian Lens matters. A concept can exist internally without appearing clearly through a direct vocabulary projection.

---

## Reference

Anthropic (2026). *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits.
https://transformer-circuits.pub/2026/workspace/index.html

---

## Disclaimer

This is an independent learning experiment. It is not an official reproduction of Anthropic's research, and the results shouldn't be treated as evidence about model consciousness or subjective experience.
