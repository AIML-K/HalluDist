# Pre-trained Language Models Return Distinguishable Probability Distributions to Unfaithfully Hallucinated Texts (EMNLP Findings 2024)

![fig1](https://github.com/AIML-K/HalluDist/blob/main/plots/fig1.png)

[Paper](https://arxiv.org/abs/2409.16658)

## Abstract

In this work, we show the pre-trained language models return distinguishable generation probability and uncertainty distribution to unfaithfully hallucinated texts, regardless of their size and structure.
By examining 24 models on 6 data sets, we find out that 88-98% of cases return statistically significantly distinguishable generation probability and uncertainty distributions.
Using this general phenomenon, we showcase a hallucination-reducing training algorithm.
Our algorithm outperforms other baselines by achieving higher faithfulness metrics while maintaining sound general text quality measures.

## Installation

1. Install Pytorch following the instruction on https://pytorch.org/get-started/locally/.
2. Install requirements with `pip3 install -r requirements.txt`.
3. Install an additional requirement with `pip3 install -U git+https://github.com/ddkang/loss_dropper.git`
4. Set your huggingface api token by `export HF_TOKEN="[YOUR TOKEN]"`. Make sure your token has access to Llama2 models.

## Run

1. To obtain the log token probability/entropy distribution, run `bash run.sh` and check `/plots/`.
2. To obtain the weighted training result follow the instruction.
- Train a benchmark CTRL model by applying the code in `https://github.com/McGill-NLP/FaithDial/tree/main/models/ctrl`.
- Run `bash run_train.sh`. (You can ignore lines 1-7 in `run.sh` and use the precomputed values in `/assets/chat_data/`)
- Obtain Q2 score by applying the code in `https://github.com/orhonovich/q-squared`.
