# RuSentEval
### Linguistic Source, Encoder Force!

RuSentEval is an evaluation toolkit for sentence embeddings for Russian.

In this repo you can find the data and scripts to run an evaluation of the quality of sentence embeddings. 

RuSentEval, an enhanced set of 14 probing tasks for Russian, including ones that have not been explored yet. We apply a combination of complementary probing methods to explore the distribution of various linguistic properties in five multilingual transformers for two typologically contrasting languages – Russian and English. 

Our results provide intriguing findings that contradict the common understanding of how linguistic knowledge is represented, and demonstrate that some properties are learned in a similar manner despite the language differences.

## Tested Models and Results

### Russian Models:
1. **RuBert** (Kuratov, Arkhipov, 2019) RuBERT by DeepPavlov (Russian, cased, 12‑layer, 768‑hidden, 12‑heads, 180M parameters) was trained on the Russian part of Wikipedia and news data. 
2. **RuBert Sentence** Sentence RuBERT by DeepPavlov (Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters) is a representation‑based sentence encoder for Russian. It is initialized with RuBERT and fine‑tuned on SNLI google-translated to russian and on russian part of XNLI dev set. Sentence representations are mean pooled token embeddings in the same manner as in Sentence‑BERT.
3. **RuBert Conversational**  Conversational RuBERT by DeepPavlov (Russian, cased, 12‑layer, 768‑hidden, 12‑heads, 180M parameters) was trained on OpenSubtitles, Dirty, Pikabu, and a Social Media segment of Taiga corpus.

### Multilingual Models:
1. **M-BERT** (Devlin et al., 2019) is trained on masked language modeling (MLM) and next sentence prediction tasks, over concatenated monolingual Wikipedia corpora in 104 languages.
1. **XLM-R** (Conneau et al., 2020a) is trained on ’dynamic’ MLM task, over filtered CommonCrawl data in 100 languages (Wenzek et al., 2020).
1. **MiniLM** (Wang et al., 2020) is a distilled transformer of BERT architecture, but uses the XLMRoBERTa tokenizer.
1. **LABSE** (Feng et al., 2020) employs a dual-encoder architecture that combines MLM and translation language modeling (Conneau and Lample, 2019).
1. **M-BART** (Liu et al., 2020) is a sequence-tosequence transformer model with a BERT encoder, and an autoregressive GPT-2 decoder (Radford et al., 2019). We used only the encoder in the experiments.


![pic1](/images/Screenshot%20from%202021-03-03%2023-16-21.png)

The probing results for each encoder on NShift (Ru) task. 
X-axis=Layer index number, Y-axis=Accuracy score.

![pic2](/images/Screenshot%20from%202021-03-03%2023-16-32.png)

The distribution of top neurons over SentLen tasks for both languages: Ru=Russian, En=English. Xaxis=Layer index number, Y-axis=Number of neurons selected from the layer.

## Tasks
The tasks are both applicable for English and Russian probing tasks, covering surface properties 
(SentLen, WC), syntax (TreeDepth, NShift), and semantics (ObjNumber, SubjNumber, and Tense). 

![pic3](/images/Screenshot%20from%202021-03-03%2023-15-47.png)

## Setup & Usage 

### Installation
TBA
### Usage
TBA

## Cite us

The paper is accepted to BSNLP workshop at EACL 2021. 
```
@misc{mikhailov2021rusenteval,
      title={RuSentEval: Linguistic Source, Encoder Force!}, 
      author={Vladislav Mikhailov and Ekaterina Taktasheva and Elina Sigdel and Ekaterina Artemova},
      year={2021},
      eprint={2103.00573},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
[arxiv link](https://arxiv.org/abs/2103.00573v2)

![paper preview](/images/Screenshot%20from%202021-03-03%2023-20-03.png)
