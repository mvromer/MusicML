---
title: "Requiem for Reproducible Research"
date: 2019-12-14T19:53:51-06:00
featured_image: "/images/requiem-hero.jpg"
draft: true
---

# Introduction
Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), the neural network architecture that have been proven to be effective in tasks that transforming sequences, like text translation. But the application of this attention-based mechanism has been extended to other fields, like gaming ([DeepMind AlphaStar](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)) and music etc.

In this project, we are trying to reproduce the implementation of the music transformer ([Huang et al. 2019](https://arxiv.org/abs/1809.04281)), it is a transformer designed and trained for generating coherent music pieces with the music snippet inputs. 

Our plan is to re-implement the model from scratch, and obtain similar results as the paper after training. Additionally, we want to compare how our implementation differs from implementations of other libraries, and make improvements on ours if possible. 

# Dataset
The dataset we use for training the testing is the [Maestro Dataset v2.0.0](https://magenta.tensorflow.org/datasets/maestro), it contains over 200 hours recordings of the International Piano-e-Competition for 10 years. Each recording was done by a Disklavier, which captures professional performers' actions and records them in a MIDI file. 

The dataset is suitable for training a machine learning model because of few reasons:
- They are all in the same music genre, classical, which helps the coherence of the output.
- They captures only piano performance. It's easier to train the model and preprocess the MIDI date with a solo instrument in the recording.
- They were performed by professional performers. Instead of training with synthesized MIDI music, human performance bring more fidelity to the dynamic of the music, it helps the models to learn and generate more expressive music.

## Preprocessing

### MIDI events conversion
Before we feed the training data to model, the dataset needs to be converted into a "digestible" format. 

The MIDI data are read and converted into a series of events, they can are grouped into 4 categories:
- NOTE_ON events: key-press event with a pitch value ranging from 0 to 127, it starts a new note.
- NOTE_OFF event: key-release event with a pitch value ranging from 0 to 127, it release a note.
- Time_SHIFT event: time step event with a time value measured in ms, it moves the time steps forward by increments of 10 ms up to 1 second.
- SET_VELOCITY event: velocity event with a velocity value ranging from 0 t0 31, it changes the velocity applied to all subsequent notes, until the next velocity event.

{{< figure src="/MusicML/images/pianoroll-vs-midi-events.png" caption="Figure 1. Comparison between a piano roll and MIDI events. ([Huang et al. 2019](https://arxiv.org/abs/1809.04281)) " >}}

### Vocabulary
After converting the MIDI data into corresponding events, the events are then map to index of the predefined vocabulary. The vocabulary has a size of 390 tokens, which includes 128 NOTE_ON and NOTE_OFF tokens (there are 128 notes on the MIDI keyboard), 100 TIME_SHIFT tokens, 32 SET_VELOCITY tokens, and a start and stop token. 

The final output of a MIDI files after preprocessing contains a list of number ranging from 0 to 389.

# Previous Work
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
including music generation. The complete Transformer architecture is depicted below.

{{< figure src="/MusicML/images/transformer.png" caption="Figure 2. The Transformer architecture" >}}

The Transformer is unique in that it comprises almost entirely of attention mechanism layers and
utilizes no recurrence or convolution. At the heart of it is an encoder stack of six encoding layers
followed by a decoder stack of six decoding layers. Each encoding layer is a combination of a
self-attention sublayer and a position-wise feed forward sublayer.

The use of self-attention plays an important role in the Transformer. It allows the model to relate
tokens in different positions of the input sequence in order to compute a representation of the
sequence and the relationships of the tokens within it with constant path length.

Similarly, each decoding layer also contains self-attention and feed forward sublayers; however, in
between these two the decoding layer also contains an encoder-decoder attention sublayer that
performs attention between the output of the encoder and the output of the self-attention sublayer.
In this arrangement, the encoder provides the decoder with contextual information it can use in the
decoding process.

As per Vaswani et al., residual connections are utilized between each sublayer. The connections sum
a sublayer's output with its input. The result is then passed through a layer normalization step
before proceeding to the next sublayer.

The output of the final decoding layer is passed through a fully connected layer that projects the
result back into the dimensionality of the output vocabulary. Finally a softmax is applied to
produce a probability distribution over the output vocabulary tokens that is then used to predict
the next output token.

In an encoder-decoder configuration of the Transformer, the decoder takes the context from the
encoder and generates the output sequence one token at a time. The Transformer is auto-regressive,
and after each decode step, the decoder feeds the previously generated token back as additional input
to the decoder for generating the next output token.

The encoder-decoder structure is designed especially for working with two different sets of
vocabularies such as what is encountered in tasks like English-French translation. In this example,
the encoder will encode the English input, and the decoder uses this context to generate the
corresponding French output.

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
triangular portion of \\((QK^T\\). The decoder's encoder-decoder attention sublayer uses the
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

# Our Contribution
Our project is based on the Music Transformer model introduced by Huang et al. We originally
intended to train this model in such a way that we could generate music progressively in three
different genres (of increasing difficulty):

* Classical
* Jazz
* Electronic Dance Music (EDM)

Similar to the Music Transformer, we figured for music generation we could utilize the Transformer's
encoder alone with auto-regression. We also based this design decision after reviewing Pytorch's
Transformer example on text generation. In their example, they also only utilize the encoder with
auto-regression when generating sequences that use the same vocabulary as the input sequence.

As will be further elaborated, our intents did not materialize as we had hoped. Instead, our project
largely turned into a replication study of us merely trying to reproduce the results of those cited
in the paper by Huang et al.

## Choice of ML Framework
Given our only real exposure to any machine learning frameworks has been that which we gained by
using Pytorch over the course of the semester, we decided to use it for implementing our Music
Transformer. Off-the-shelf, Pytorch provides an implementation of the Transformer as described by
Vaswani et al. However, it provided no support for using relative position encodings within its
self-attention mechanism, so we were prepared to implement that feature ourselves in our model.

Pytorch's Transformer module consists of a number of layered components. The `Transformer` component
encapsulates the `TransformerEncoder` and a `TransformerDecoder` components. These respectively
represent the stack of `TransformerEncoderLayer` and `TransformerDecoderLayer` components, each
implementing the encoding and decoding sublayers.

Their design is such that you can tap in at different levels to provide custom implementations,
which on face value sounded great. All we needed to do was tap in a provide a custom attention
mechanism that incorporated the relative position information in its calculation, and we could
leverage the rest of Pytorch's Transformer implementation for free.

However, this is when we realized that the `TransformerEncoderLayer` and `TransformerDecoderLayer`
were tightly coupled to Pytorch's `MultiheadAttention` component. Unlike the various other
components and settings within their Transformer implementation, the multihead attention mechanism
was **not** something for which we could provide an alternate implementation. This literally
meant reimplementing the entire `TransformerEncoderLayer` component, including the feed-forward
sublayer and residual connections.

Since the `TransformerEncoder` component encapsulated just a list of `TransformerEncoderLayer`
components, we saw little value in actually using any of the Pytorch abstraction at this point.
Using Pytorch's Transformer implementation as guidance as well as resources like
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and
[How to Code the Transformer in Pytorch](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec),
we proceeded to implement our own version of the Transformer suitable for replicating the Music
Transformer architecture.

## Implementation
We first built out the custom multihead attention mechanism. We made it possible to toggle between
the standard multihead attention used by Vaswani et al. and a variation that computes the relative
logits using Huang et al.'s skewing procedure and uses them to compute the final attention value.

# Training

# Results
After training the baby batch for 10 epochs of 48350 steps each using the same set of hyperparameters, the results are as follows:

|                             | Ours (Absolute Only) | Ours (Absolute + Relative) | Pytorch Transformer-based |
| --------------------------- |:--------------------:| --------------------------:| ------------------------: |
| Average training loss (NLL) | 4.006                | 4.0064                     | 4.0059                    |

All three models gave average training loss at around 4.00, the relative positional attention didn't show any obvious improvements against the implementation with absolute positioning alone. Moreover, from the result of the Pytorch transformer-based implementation, we can probably make an early conclusion that our implementation is similar to the Pytorch one. 

# Samples
