# RuSentEval
### Linguistic Source, Encoder Force!

RuSentEval is an evaluation toolkit for sentence embeddings for Russian.

In this repo you can find the data and scripts to run an evaluation of the quality of sentence embeddings. 

RuSentEval, an enhanced set of 14 probing tasks for Russian, including ones that have not been explored yet. We apply a combination of complementary probing methods to explore the distribution of various linguistic properties in five multilingual transformers for two typologically contrasting languages – Russian and English. 

Our results provide intriguing findings that contradict the common understanding of how linguistic knowledge is represented, and demonstrate that some properties are learned in a similar manner despite the language differences.


## Probing Tasks
The main body of paper presents the results for English and Russian tasks, covering surface properties 
(SentLen, WC), syntax (TreeDepth, NShift), and semantics (ObjNumber, SubjNumber, and Tense). 

![pic3](/images/Screenshot%20from%202021-03-03%2023-15-47.png)


## Data
We publicly release the [sentences](https://disk.yandex.ru/d/78FfMVzLPECteQ) that were used to construct the probing tasks. The sentences were annotated with the current [SOTA](https://github.com/DanAnastasyev/GramEval2020) model for joint morphosyntactic analysis for Russian. The total number of sentences is 3.6kk. The filtering procedure is described in our paper.

## Probing Methods
* [Supervised probing](https://github.com/RussianNLP/rusenteval/tree/main/probing) involves training a Logistic Regression classifier to predict a property. The performance is used as a proxy to evaluate the model knowledge.
* [Neuron-level Analysis](https://github.com/fdalvi/NeuroX) [Durrani et al., 2020] allows retrieving a group of individual neurons that are most relevant to predict a linguistic property.
* [Contextual Correlation Analysis](https://github.com/johnmwu/contextual-corr-analysis/tree/master) [Wu et al., 2020] is a representation-level similarity measure that allows identifying pairs of layers of similar behavior. 

## Tested Models and Results
The code is compatible with models released as a part of HuggingFace library.

### Russian Models:
1. **RuBERT** (Kuratov, Arkhipov, 2019) RuBERT by DeepPavlov (Russian, cased, 12‑layer, 768‑hidden, 12‑heads, 180M parameters) was trained on the Russian part of Wikipedia and news data. 
2. **Sentence RuBERT** Sentence RuBERT by DeepPavlov (Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters) is a representation‑based sentence encoder for Russian. It is initialized with RuBERT and fine‑tuned on SNLI google-translated to russian and on russian part of XNLI dev set. Sentence representations are mean pooled token embeddings in the same manner as in Sentence‑BERT.
3. **Conversational RuBERT**  Conversational RuBERT by DeepPavlov (Russian, cased, 12‑layer, 768‑hidden, 12‑heads, 180M parameters) was trained on OpenSubtitles, Dirty, Pikabu, and a Social Media segment of Taiga corpus.

### Multilingual Models:
1. **M-BERT** (Devlin et al., 2019) is trained on masked language modeling (MLM) and next sentence prediction tasks, over concatenated monolingual Wikipedia corpora in 104 languages.
2. **XLM-R** (Conneau et al., 2020a) is trained on ’dynamic’ MLM task, over filtered CommonCrawl data in 100 languages (Wenzek et al., 2020).
3. **MiniLM** (Wang et al., 2020) is a distilled transformer of BERT architecture, but uses the XLMRoBERTa tokenizer.
4. **LABSE** (Feng et al., 2020) employs a dual-encoder architecture that combines MLM and translation language modeling (Conneau and Lample, 2019).
5. **M-BART** (Liu et al., 2020) is a sequence-tosequence transformer model with a BERT encoder, and an autoregressive GPT-2 decoder (Radford et al., 2019). We used only the encoder in the experiments.


![pic1](/images/Screenshot%20from%202021-03-03%2023-16-21.png)

The probing results for each encoder on NShift (Ru) task. 
X-axis=Layer index number, Y-axis=Accuracy score.

![pic2](/images/Screenshot%20from%202021-03-03%2023-16-32.png)

The distribution of top neurons over SentLen tasks for both languages: Ru=Russian, En=English. Xaxis=Layer index number, Y-axis=Number of neurons selected from the layer.

## Setup & Usage 
### Installation
```
git clone https://github.com/RussianNLP/rusenteval/
cd rusenteval
sh install_tools.sh
```
### Usage
#### Example: Baseline
```
from probing.arguments import ProbingArguments
from probing.baseline import Baseline


args = ProbingArguments()

# args.clf == "logreg" or args.clf == "mlp" for linear/non-linear classification
args.clf = "logreg"

# for imbalanced datasets (gapping, tree_depth)
args.balanced = False

tasks = ["gapping"]

# define the baseline features to run
baseline = Baseline(args, tasks, baseline_features=["tfidf_word"])
baseline.run()
```


#### Example: Experiment (Supervised probing)
```
from probing.arguments import ProbingArguments
from probing.experiment import Experiment


tasks = ["sent_len", "subj_number"]
# name of the HuggingFace model; you can adjust the code for your model
model = "bert-base-multilingual-cased"
args = ProbingArguments()

# args.clf == "logreg" or args.clf == "mlp" for linear/non-linear classification
args.clf = "mlp"

experiment = Experiment(tasks, model, args)
experiment.run()
```

## Cite us
The [paper](https://arxiv.org/abs/2103.00573v2) is accepted to BSNLP workshop at EACL 2021. The title follows Power Rangers Mystic Force series (Roll Call Team-Morph: "Magical Source, Mystic Force!")

```
@inproceedings{mikhailov-etal-2021-rusenteval,
    title = "{R}u{S}ent{E}val: Linguistic Source, Encoder Force!",
    author = "Mikhailov, Vladislav  and
      Taktasheva, Ekaterina  and
      Sigdel, Elina  and
      Artemova, Ekaterina",
    booktitle = "Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.bsnlp-1.6",
    pages = "43--65",
    abstract = "The success of pre-trained transformer language models has brought a great deal of interest on how these models work, and what they learn about language. However, prior research in the field is mainly devoted to English, and little is known regarding other languages. To this end, we introduce RuSentEval, an enhanced set of 14 probing tasks for Russian, including ones that have not been explored yet. We apply a combination of complementary probing methods to explore the distribution of various linguistic properties in five multilingual transformers for two typologically contrasting languages {--} Russian and English. Our results provide intriguing findings that contradict the common understanding of how linguistic knowledge is represented, and demonstrate that some properties are learned in a similar manner despite the language differences.",
}
```
