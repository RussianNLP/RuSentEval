import warnings

warnings.simplefilter(action="ignore", category=(FutureWarning, DeprecationWarning))

import os
import gc
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from fasttext import load_model, util
from torch.autograd import Variable
from torch.nn.modules.sparse import EmbeddingBag

from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, XLMRobertaTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from probing.prober import Prober
from probing.arguments import ProbingArguments
from probing.utils import save_results, singleton


@singleton
class FastTextVectorizer(object):
    def __init__(
        self,
        model_dir: str = "models",
        fasttext_model_name: str = "ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin",
        fasttext_en_model_name : str = "cc.en.300.bin",
    ):
        self.model_dir = model_dir
        self.fasttext_model_name = fasttext_model_name
        self.fasttext_en_model_name = fasttext_en_model_name
        self.model_path = os.path.join(self.model_dir, self.fasttext_model_name)
        if self.fasttext_model_name == "en":
            util.download_model('en', if_exists='ignore')
            self.model = load_model(self.fasttext_en_model_name)
        else:
            self.model = load_model(self.model_path)
        self.input_matrix = torch.FloatTensor(self.model.get_input_matrix())
        self.matrix_shape = self.input_matrix.shape
        self.embedding_bag = EmbeddingBag(
            self.matrix_shape[0], self.matrix_shape[1]
        ).from_pretrained(self.input_matrix, mode="mean").cuda()

    def forward(self, x):
        token_offsets = [0]
        token_subindexes = torch.empty([0], dtype=torch.int64)
        for token in x:
            _, subinds = self.model.get_subwords(token)
            token_subindexes = torch.cat((token_subindexes, torch.from_numpy(subinds)))
            token_offsets.append(token_offsets[-1] + len(subinds))
        token_offsets = token_offsets[:-1]
        ind = Variable(torch.LongTensor(token_subindexes)).cuda()
        offsets = Variable(torch.LongTensor(token_offsets)).cuda()
        return self.embedding_bag(ind, offsets)


@singleton
class BaselineFeaturizer(object):
    def __init__(
        self,
        fasttext_model_name: str,
        bert_model_name: str,
        xlmr_model_name: str,
    ):
        self.fasttext_model_name = fasttext_model_name
        self.bert_model_name = bert_model_name
        self.xlmr_model_name = xlmr_model_name
        self.fasttext_vectorizer = FastTextVectorizer(
            fasttext_model_name=self.fasttext_model_name
        )
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.bert_model_name, do_lower_case=False
        )
        self.xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained(
            self.xlmr_model_name, do_lower_case=False
        )
        self.word_vectorizer = TfidfVectorizer(tokenizer=self.tokenize, lowercase=False, min_df=0.1, max_features=250000)
        self.char_vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            lowercase=False,
            analyzer="char",
            ngram_range=(1, 4),
            min_df = 0.1,
            max_features=250000
        )
        self.bert_vectorizer = TfidfVectorizer(
            tokenizer=self.bert_tokenizer.tokenize, lowercase=False, ngram_range=(1, 4), min_df=0.1, max_features=250000
        )
        self.xlmr_vectorizer = TfidfVectorizer(
            tokenizer=self.xlmr_tokenizer.tokenize, lowercase=False, ngram_range=(1, 4), min_df=0.1, max_features=250000
        )

    def tokenize(self, sentence):
        return sentence.split()

    def create_tfidf_vectors(
        self,
        df: pd.DataFrame,
        vectorizer: TfidfVectorizer,
        subset: str,
        df_field: str = "sentence",
    ):
        vectors = (
            vectorizer.fit_transform(df[df_field])
            if subset == "tr"
            else vectorizer.transform(df[df_field])
        )
        return vectors.toarray()

    def create_char_num_vectors(self, df: pd.DataFrame, df_field: str = "sentence"):
        vectors = np.array([len(w) for w in df[df_field]]).reshape(-1, 1)
        return vectors

    def create_distributed_vectors(
        self,
        df: pd.DataFrame,
        vectorizer: FastTextVectorizer,
        df_field: str = "sentence",
    ):
        fasttext = lambda x: np.mean(np.vstack(vectorizer.forward(self.tokenize(x)).cpu().numpy()), 0)
        vectors = [
            fasttext(x) for x in tqdm(df[df_field], total=df.shape[0])
        ]
        vectors = np.vstack(vectors)
        return vectors


class Baseline(object):
    def __init__(
        self,
        args: ProbingArguments,
        probe_tasks: list,
        baseline_features: list = [
            "fasttext",
            "char_number",
            "tfidf_word",
            "tfidf_char_ngrams",
            "tfidf_bpe_tokens",
            "tfidf_sentencepiece_tokens",
        ],
        fasttext_model_name: str = "ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin",
        bert_model_name: str = "bert-base-multilingual-cased",
        xlmr_model_name: str = "xlm-roberta-base",
        result_dir: str = "baseline_results",
    ):
        self.args = args
        self.probe_tasks = probe_tasks
        self.baseline_features = baseline_features
        self.fasttext_model_name = fasttext_model_name
        self.bert_model_name = bert_model_name
        self.xlmr_model_name = xlmr_model_name
        self.result_dir = result_dir
        self.featurizer = BaselineFeaturizer(
            fasttext_model_name=self.fasttext_model_name,
            bert_model_name=self.bert_model_name,
            xlmr_model_name=self.xlmr_model_name,
        )

    def prepare_from_dataframe(
        self, probe_task: str, subsets: list, data_dir: str = "data"
    ):
        df = pd.read_csv(
            os.path.join(os.getcwd(), data_dir, probe_task) + ".txt",
            sep="\t",
            names=("subset", "label", "sentence"),
        )
        df["index"] = [i for i in range(df.shape[0])]
        le = LabelEncoder()
        task_dataframe = {subset: df[df["subset"] == subset] for subset in subsets}
        for subset, subset_dataframe in task_dataframe.items():
            subset_dataframe["label"] = (
                le.fit_transform(subset_dataframe["label"])
                if subset == "tr"
                else le.transform(subset_dataframe["label"])
            )
        _label_encoder = {
            str(label): int(le.transform([label]).item()) for label in le.classes_
        }
        return task_dataframe, _label_encoder

    def create_features(
        self,
        probe_task: str,
        feature_name: str,
        df_field: str = "sentence",
        subsets: list = ["tr", "va", "te"],
    ):
        vector_features = {
            "fasttext": lambda df, subset: self.featurizer.create_distributed_vectors(
                df=df, vectorizer=self.featurizer.fasttext_vectorizer, df_field=df_field
            ),
            "char_number": lambda df, subset: self.featurizer.create_char_num_vectors(
                df=df, df_field=df_field
            ),
            "tfidf_word": lambda df, subset: self.featurizer.create_tfidf_vectors(
                df=df,
                vectorizer=self.featurizer.word_vectorizer,
                subset=subset,
                df_field=df_field,
            ),
            "tfidf_char_ngrams": lambda df, subset: self.featurizer.create_tfidf_vectors(
                df=df,
                vectorizer=self.featurizer.char_vectorizer,
                subset=subset,
                df_field=df_field,
            ),
            "tfidf_bpe_tokens": lambda df, subset: self.featurizer.create_tfidf_vectors(
                df=df,
                vectorizer=self.featurizer.bert_vectorizer,
                subset=subset,
                df_field=df_field,
            ),
            "tfidf_sentencepiece_tokens": lambda df, subset: self.featurizer.create_tfidf_vectors(
                df=df,
                vectorizer=self.featurizer.xlmr_vectorizer,
                subset=subset,
                df_field=df_field,
            ),
        }

        input_dim = None
        features = {subset: [] for subset in subsets}
        task_dataframe, _label_encoder = self.prepare_from_dataframe(
            probe_task=probe_task, subsets=subsets
        )

        for subset, subset_dataframe in task_dataframe.items():
            for index, label, feature_vector in zip(
                subset_dataframe["index"].values,
                subset_dataframe["label"].values,
                vector_features[feature_name](subset_dataframe, subset),
            ):
                index_tensor = torch.tensor(index)
                feature_vector_tensor = torch.from_numpy(feature_vector).float()
                label_tensor = torch.tensor(label)
                features[subset].append(
                    (index_tensor, feature_vector_tensor, label_tensor)
                )

                if input_dim is None:
                    input_dim = feature_vector_tensor.size(0)
        self.args.input_dim = input_dim
        return features, _label_encoder

    def run(self):
        print(f"Running baseline for the following tasks: {self.probe_tasks}")

        for probe_task in self.probe_tasks:
            print(f"Baseline: the {probe_task} task...")
            for feature_name in self.baseline_features:
                baseline_task_results = {}
                baseline_task_results["probe_task"] = probe_task
                print(f"Feature: {feature_name}")
                features, _label_encoder = self.create_features(
                    probe_task=probe_task, feature_name=feature_name
                )
                baseline_prober = Prober(
                    args=self.args,
                    probe_task=probe_task,
                    model_name=feature_name,
                    features=features,
                )
                baseline_prober.args.num_classes = len(_label_encoder)
                baseline_prober.hdf5handler.label_encoder.update(_label_encoder)
                baseline_task_result = baseline_prober.probe()
                baseline_task_results[feature_name] = baseline_task_result
                baseline_task_results["baseline_arguments"] = baseline_prober.args.__dict__

                save_results(
                    probe_task,
                    baseline_task_results,
                    feature_name,
                    self.args.clf,
                    self.result_dir,
                )
                
                del features, _label_encoder, baseline_prober
                
                gc.collect()
