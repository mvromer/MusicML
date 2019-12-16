---
title: "Requiem for Reproducible Research"
date: 2019-12-14T19:53:51-06:00
featured_image: "/images/requiem-hero.jpg"
draft: true
---

# Introduction

# Dataset

# Background
Below we go over each of the key concepts on which our project was based. We begin looking closely
at the Transformer architecture ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) followed
by discussing some of the contributions made by others
([Shaw et al. 2018](https://arxiv.org/abs/1803.02155) and
[Huang et al. 2018](https://arxiv.org/abs/1809.04281)) that are of benefit to the task of music
generation.

## Transformer
The Transformer is a modern-day neural network architecture that has inspired many other designs. At
the time of its publication, it was the highest performing model for sequence-to-sequence machine
translation tasks, outperforming even contemporary recurring neural networks considered then as
state of the art. Since then, the Transformer has been applied to many other non-language domains
including music generation.

The Transformer is unique in that it comprises almost entirely of attention mechanism layers and
utilizes no recurrence or convolution. At the heart of it is an encoder stack of six encoding layers
followed by a decoder stack of six decoding layers. Each encoding layer is a combination of a
self-attention sublayer and a position-wise feed forward sublayer. Similarly, each decoding layer
also contains self-attention and feed forward sublayers; however, in between these two the decoding
layer also contains an additional sublayer that performs attention between the output of the encoder
and the output of the self-attention sublayer.

As per Vaswani et al.,, residual connections are utilized between each sublayer. The connections sum
a sublayer's output with its input. The result is then passed through a layer normalization step
before proceeding to the next sublayer.

The output of the final decoding layer is passed through a fully connected layer that projects the
result back into the dimensionality of the output vocabulary. Finally a softmax is applied to
produce a probability distribution over the output vocabulary tokens that is then used to predict
the next output token.

## Input Embedding
In language tasks, the input is usually a sequence of words. In music, it can be a sequence of pitch
and timing events. In order for a neural network to properly process these sequences, it's necessary
to represent each word/event as a token from some predetermined vocabulary.

Each token is represented by a multidimensional vector that the network can process. An embedding
layers performs the job of converting input tokens into these *embedding vectors*. The embeddings
can either be learned as a byproduct of the model's training process or predetermined, such as a
one-hot encoding. Here we'll refer to the dimensionality of the embedding vector as \\(d_E\\).

## Attention Mechanism
Below we will briefly give an overview of the attention mechanism found in the Transformer. For a
more in-depth description, we recommend reading the excellent article
[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/).

An attention mechanism projects a \\(d_E\\)-dimensional embedding vector into a *key* vector
subspace, a *query* vector subspace, and a *value* vector subspace. The dimensionality of the key
and query subspaces is \\(d_K\\), and the dimensionality of the value subspace is \\(d_V\\).

The weight matrices \\(W_K\\), \\(W_Q\\), and \\(W_V\\) used to transform each embedding vector into
the corresponding key, query, and value subspace become learnable parameters of the model. In
practice, \\(L\\) embedding vectors are stacked on top of each other to form an \\(L\\)-by-\\(d_E\\)
matrix \\(X\\). From this the key, query, and value matrices are formed by three matrix
multiplications: \\(K=XW_K\\), \\(Q=XW_Q\\), and \\(V=XW_V\\).

Multiple attention values are computed in parallel through additional matrix level operations,
including matrix multiplications. The final attention value is given by the following equation:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\lparen \frac{QK^T}{\sqrt{d_K}} \rparen V
$$

The Transformer's attention mechanism projects the matrix \\(X\\) to key, vector, and query
subspaces \\(H\\) times, where \\(H\\) is the number of attention heads used by the Transformer.
The attention values for each head are concatenated together and linearly projected back to the
embedding vector space by a matrix \\(W_O\\), which also becomes a learnable parameter of the model.

The self-attention sublayers in the Transformer use the incoming embedding vectors to compute
\\(K\\), \\(Q\\), and \\(V\\). Additionally, the decoder's self-attention will mask out the upper
triangular portion of \((QK^T\\). The decoder's encoder-decoder attention sublayer uses the
encoder's output to compute \\(K\\) and \\(Q\\) and uses the output of its self-attention to compute
\\(V\\).

## Encoding Positional Information
The Transformer itself is invariant to the position of tokens in the input sequence. This is in
contrast to other model types like RNNs and LSTMs whose model representations directly support
capturing the ordering of tokens within a sequence.

As an example, in the sentence *Do you want chocolate pie or apple pie?* the word *pie* would be
represented by the same embedding vector. Thus, both occurrences of the word *pie* would be treated
equivalently by the Transformer. However, their positional information within the sentences makes
clear that each occurrence refers to a different concept and thus should be handled distinctly.

Vaswani et al. resolved this issue by encoding absolute positional information for each token in an
input sequence. Their idea used sinusoids to build a \\(d_E\\)-dimensional absolute position
representation for each input token. These positional vectors were then summed with their
corresponding input embedding vectors. Thus, the same input token occurring at different locations
within the input sequence would have different representations within the Transformer model.

Instead of encoding absolute positions, Shaw et al. showed how to instead encode *relative*
positional information within the self-attention mechanism of the Transformer. This required the
formation of an intermediate tensor \\(R\\) with dimensions \\(L\\)-by-\\(L\\)-by-\\(d_E\\) that
encoded the pairwise relative position information for each input token. The so-called
*relative logits* were then formed by computing \\(QR^T\\) which were then used to modify the
attention calculation in the following manner:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\lparen \frac{QK^T + QR^T}{\sqrt{d_K}} \rparen V
$$

## Music Transformer
For the generation of long sequences, \\(R\\)'s \\(O(L^2D)\\) space complexity becomes the
dominating factor in terms of memory consumption. Given the constraints of contemporary hardware,
lengthy sequences became impossible to realize.

Huang et al.'s proposed a "skewing" procedure that enabled the computation of \\(QR^T\\) without the
explicit formation of \\(R\\). Instead, they used an alternate relative position encoding matrix
\\(E_r\\) with size \\(L\\)-by-\\(D_E\\). They then compute \\(\mathrm{skew}(QE_r^T)\\) to form the
relative logits. Thus, under Huang et al. the self-attention calculation with relative position
encoding becomes the following:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\lparen \frac{QK^T + \mathrm{skew}(QE_r^T)}{\sqrt{d_K}} \rparen V
$$

Using Huang et al.'s formulation, the space complexity goes from being quadratic in the length of
the generated sequence to being linear. This enabled them to develop the Music Transformer, which
has demonstrated state-of-the-art capability of generating musical sequences exhibiting long-term
structure. Their model allowed them create sequences with a maximal length of 3500 tokens in their
symbolic music representation.

# Models
Our project is based on the Music Transformer model introduced by Huang et al. We originally
intended to train this model in such a way that we could generate music progressively in three
different genres (of increasing difficulty):

* Classical
* Jazz
* Electronic Dance Music (EDM)

# Training

# Results

# Samples
