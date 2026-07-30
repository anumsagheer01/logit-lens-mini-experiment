<p align="center">
  <img src="./assets/anthropic-header.svg" alt="Looking Inside Language Models" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-D97757?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TransformerLens-Interpretability-F7F4ED?style=for-the-badge&labelColor=191919" alt="TransformerLens">
  <img src="https://img.shields.io/badge/Research-Logit%20Lens-191919?style=for-the-badge" alt="Logit Lens">
</p>

Main question: Can a language model represent a concept internally even when that concept does not appear in its final output?

Overview

This project is a small mechanistic interpretability experiment that looks at how possible next-token predictions change across transformer layers.

It was inspired by Anthropic's paper "Verbalizable Representations Form a Global Workspace in Language Models." The paper studies whether language models contain internal representations that can be used in reasoning and later expressed in words.

Anthropic introduces the Jacobian Lens, also called the J-lens, to study these representations. The paper refers to the relevant internal region as J-space.

This notebook does not reproduce the full Jacobian Lens method. It uses the simpler logit lens technique to inspect the model's residual stream after each layer.

The Experiment

The models received this prompt:

Concentrate on the concept of oranges.
Now copy this sentence exactly:
The old painting hung crookedly on the

The prompt introduces two different ideas:

A hidden concept to concentrate on: oranges

A sentence that should likely end with: wall

The experiment checks whether orange-related tokens appear in the model's intermediate predictions, even though the final sentence is unrelated to oranges.

Models Tested

Model

Approximate size

Model type

GPT-2

124M parameters

Base language model

Qwen1.5-0.5B-Chat

500M parameters

Instruction-tuned chat model

Qwen1.5-1.8B-Chat

1.8B parameters

Instruction-tuned chat model

All models were loaded through TransformerLens.

How the Logit Lens Works

A transformer stores and updates information in a shared internal vector called the residual stream.

The notebook takes the residual stream after each layer and passes it through the model's final normalization and unembedding matrix.

Intermediate residual stream
            ↓
Final layer normalization
            ↓
Unembedding matrix
            ↓
Vocabulary token scores

This gives a rough view of which tokens each layer appears to support.

The main operation is:

resid = cache["resid_post", layer][0, last_pos]
resid_normed = model.ln_final(resid)
layer_logits = resid_normed @ model.W_U

The notebook then records the five highest-scoring tokens at every layer.

Logit Lens and Jacobian Lens

Logit Lens

The logit lens directly projects an intermediate activation into vocabulary space using the model's final output projection.

It is simple and useful for seeing how next-token predictions develop across layers.

However, early and late transformer layers may use different internal coordinate systems. Because of this, early-layer logit lens results can appear noisy or misleading.

Jacobian Lens

Anthropic's Jacobian Lens estimates how an earlier activation can affect current and future outputs after passing through the remaining layers.

This helps account for changes in representation across model depth.

The two methods are related, but they are not equivalent. This project should be understood as a smaller experiment inspired by the paper's questions, not a reproduction of its full method.

Results

GPT-2

GPT-2 showed the clearest progression across layers.

Early layers produced broad or uncertain tokens:

same
latter
simplest
smallest

Middle layers began moving toward physical surfaces and locations:

edges
horizon
shoulders
walls
floor

Later layers strongly supported the expected sentence ending:

wall
walls
floor
porch
fireplace
ceiling

By the final layer, wall was the strongest prediction.

Qwen1.5-0.5B-Chat

The smaller Qwen model produced noisy and multilingual tokens in many early and middle layers.

Near the final layers, its predictions became more connected to the copying task:

The
the
wall
old

This suggests that its hidden state became easier to interpret as it moved closer to the final output.

Qwen1.5-1.8B-Chat

The larger Qwen model also produced noisy early-layer predictions.

From around layer 16 onward, its predictions became more stable and focused on the start of the copied sentence:

The
the
old

Unlike GPT-2, the final token position was strongly influenced by the chat template and generation formatting.

Main Finding

No clearly orange-related token appeared among the top five predictions for any of the three models.

The models instead became increasingly focused on the sentence they were expected to produce.

This does not prove that the models completely ignored the concept of oranges.

It only shows that orange-related information was not visible:

through this specific logit lens readout

at the final prompt position

among only the top five vocabulary predictions

for this single prompt

The concept may still have been represented in another form, at another position, or below the top five token rankings.

Connection to Anthropic's Paper

The Anthropic paper studies verbalizable representations, meaning internal representations that a model may be able to express in words or use during reasoning.

The paper proposes that some of these representations form a sparse internal region called J-space. It connects this idea to a global workspace, where selected information becomes available to different parts of a system.

Important terms from the paper include:

Verbalizable representation: An internal concept that may be available for the model to express in language.

J-space: The paper's name for a sparse region containing verbalizable representations.

Global workspace: A shared space where selected information can become available for flexible use.

Jacobian Lens: A method for estimating how internal activations influence current and future outputs.

Directed attention: A model's ability to focus on a concept based on an instruction.

Causal intervention: Changing an internal activation to test whether it affects model behavior.

This notebook explores a related question in a simpler way. It asks whether a directed concept, such as oranges, becomes visible in intermediate token predictions.

What the Experiment Shows

The clearest result is not hidden orange information. It is the gradual formation of the final prediction.

For GPT-2, the progression looked roughly like this:

Broad token possibilities
          ↓
Physical surfaces and locations
          ↓
Sentence-relevant objects
          ↓
wall

This gives a simple view of how a model's internal state can become more aligned with its next-token prediction as information moves through the network.

The Qwen results also show an important limitation of the basic logit lens. Earlier layers can produce fragmented, multilingual, or unrelated tokens because those activations may not yet align with the final output space.

Limitations

This is not a full Jacobian Lens implementation

The notebook directly applies the final normalization and unembedding matrix to intermediate activations. Anthropic's method accounts for downstream transformations using Jacobians.

Only the top five tokens were inspected

An orange-related token may have appeared at a lower rank.

Only one prompt was tested

A single prompt is not enough to establish a general pattern.

Only the final prompt position was analyzed

The orange concept may be stronger near the part of the prompt where it was introduced.

Tokenization may hide related concepts

Different models may split words such as orange,  oranges, fruit, or citrus into different token pieces.

Chat templates affect the comparison

GPT-2 received the raw prompt. The Qwen chat models received extra formatting tokens from their chat templates.

No causal test was performed

The notebook reads activations but does not edit, remove, or inject them.

Possible Next Steps

A stronger follow-up experiment could:

track the exact rank of orange, oranges, fruit, and citrus at every layer

inspect more than the top five tokens

compare the same sentence with and without the orange instruction

inspect every token position in the prompt

test many concepts and sentence types

compare base models with instruction-tuned models

plot target-token probability across layers

use a tuned lens or Jacobian Lens

add activation steering or ablation experiments

test whether the hidden concept changes later generated text

Project Structure

.
├── logit_lens_experiment.ipynb
└── README.md

Installation

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Activate it on macOS or Linux:

source venv/bin/activate

Install the required packages:

pip install torch transformer-lens jupyter

Run the Notebook

Start Jupyter Notebook:

jupyter notebook

Then open:

logit_lens_experiment.ipynb

Run the cells in order.

The models are downloaded from Hugging Face during the first run. The Qwen1.5-1.8B model requires significantly more memory than GPT-2 and Qwen1.5-0.5B.

Key Terms

Term

Simple meaning

Mechanistic interpretability

Studying how a model's internal components produce its behavior

Activation

A numerical representation created inside the model

Residual stream

The shared internal vector that carries information through transformer layers

Logits

Raw scores for possible output tokens

Unembedding matrix

The matrix that converts hidden states into vocabulary scores

Logit lens

A method for reading intermediate activations as token predictions

Jacobian Lens

A method for estimating how intermediate activations affect later outputs

J-space

The paper's proposed region of verbalizable internal representations

Global workspace

A shared space where selected information can be used across processes

Activation steering

Changing internal activations to influence model behavior

Ablation

Removing or suppressing an internal component to test its role

Takeaway

This experiment did not find a clear orange-related token in the top logit lens predictions.

It did show how token predictions change across transformer layers and how later layers become increasingly aligned with the final output.

It also highlights why the distinction between the logit lens and Anthropic's Jacobian Lens matters. A concept may exist internally without appearing clearly through a direct vocabulary projection.

Reference

Verbalizable Representations Form a Global Workspace in Language ModelsAnthropic, Transformer Circuits, 2026https://transformer-circuits.pub/2026/workspace/index.html

Disclaimer

This is an independent learning experiment. It is not an official reproduction of Anthropic's research, and the results should not be treated as evidence about model consciousness or subjective experience.
