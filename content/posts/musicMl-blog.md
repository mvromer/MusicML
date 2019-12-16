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


# Transformer
The transformer architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) was introduced in 2017, aimed to improve performance and output quality from the existing sequence transduction models that based on complex recurrent neural network and convolutional neural networks. Self-attention plays an important role in the transformer to achieve such success, this attention mechanism relates different positions of a single sequence in order to compute a representation of the sequence. In fact, the transformer is the first transduction model relaying solely on self-attention to compute representations of the input and output. 

The transformer adopts the encoder-decoder structure similar to other competitive neural sequence transduction models. The encoder maps an input sequence to a sequence of continuous representation, it is the context output of the encoder and relates the input tokens of different position. The decoder takes the context from the encoder and generate the output sequence one at a time, auto-regression happens at each step, which feeds the previously generated sequence as additional input to the decoder for generating the next output sequence. 

{{< figure src="/MusicML/images/transformer.png" caption="Figure 2. The transfofmer architecture" >}}

The encoder-decoder structure is designed for working with two different sets of vocabularies, tasks like English-French translation, the encoder interpret the english input and the decoder reads the context and output the corresponding French output. We figured for music generation, it is not necessary for the decoder to generate the sequences of the input vocabulary, this can be done by the encoder alone with the auto-regression, thus, we didn't utilize the decoder in our implementation. We also confirmed that in the sequence generation example of Pytorch that decoder was absent in the implementation as well.

## Encode

## Multi-head Attention

## Relative Position Embedding

# Training

# Results

# Samples