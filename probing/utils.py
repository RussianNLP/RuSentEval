import os
import gc

import json
import h5py
import math

import torch
import random
import numpy as np

from typing import Union
from tqdm.auto import tqdm
from functools import wraps

from probing.arguments import ProbingArguments
from probing.modeling_utils import TransformerModel, MeanMaskedPooling

from torch.utils.data import Dataset, DataLoader


def init_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance

    return inner


def save_results(
    probe_task: str,
    obj: Union[list, dict],
    model_name: str,
    clf_name: str,
    save_dir: str = "bootstrap",
):
    """
    A function that saves scores or results for a given layer of the transformer model on a probe task
    :param probe_task: The probe task name
    :param obj: A JSON-serializable object to save
    :param model_name: The transformer model name
    :param clf_name: The classifier name (logreg or mlp) if given
    :param save_dir: The directory name to store layer features or scores
    """
    result_dir_path = os.path.join(os.getcwd(), save_dir, model_name)
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)

    result_fname_path = os.path.join(result_dir_path, f"{probe_task}_{clf_name}.json")
    with open(result_fname_path, "w+", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def create_saving_directory(
    probe_task: str, model_name: str, save_dir_name: str
) -> str:
    """
    A function that creates a directory to store features or scores for each layer of the transformer model on a probe task
    :param probe_task: The probe task name
    :param model_name: The transformer model name
    :param save_dir: The directory name to store layer features or scores
    :return: The total path to store layer features or scores
    """
    probe_task_dir_path = os.path.join(os.getcwd(), save_dir_name, probe_task)
    model_name = model_name.replace("/", "-")
    probe_task_model_dir_path = os.path.join(probe_task_dir_path, model_name)

    if not os.path.exists(probe_task_dir_path):
        os.makedirs(probe_task_dir_path)

    if not os.path.exists(probe_task_model_dir_path):
        os.makedirs(probe_task_model_dir_path)

    return probe_task_model_dir_path


class FeatureDataset(Dataset):
    def __init__(self, features):
        """
        Dataset object for the features (example index, hidden_state, target_label)
        :param features: A list of tuples (example index, hidden state, target label) loaded from a .h5 file
        """
        super(FeatureDataset, self).__init__()
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class ProbingDataset(Dataset):
    def __init__(self, probe_task: str, prepro_batch_size: int, bucketing: bool):
        """
        Dataset object for creating features for the probe task
        :param probe_task: The probe task name
        :param prepro_batch_size: The size of the batch for preprocessing
        :param bucketing: Whether to perform char-level sequence bucketing
        """
        super(ProbingDataset, self).__init__()
        self.probe_task = probe_task
        self.prepro_batch_size = prepro_batch_size
        self.bucketing = bucketing
        self.examples = self.load_dataset(
            probe_task=self.probe_task,
            prepro_batch_size=self.prepro_batch_size,
            bucketing=self.bucketing,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def load_dataset(
        probe_task: str, prepro_batch_size: int, bucketing: bool, data_dir: str = "data"
    ) -> list:
        examples = []
        ind = 0
        with open(
            os.path.join(os.getcwd(), data_dir, probe_task) + ".txt",
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                subset, label, sentence = line.strip().split("\t")
                examples.append((ind, subset, label, sentence, len(sentence.split())))
                ind += 1

        if bucketing:
            batches = []
            examples = sorted(examples, key=lambda x: x[-1], reverse=True)
            for i_batch in range(math.ceil(len(examples) / prepro_batch_size)):
                batches.extend(
                    examples[
                        i_batch * prepro_batch_size : (i_batch + 1) * prepro_batch_size
                    ]
                )
            return batches

        return examples


class HDF5Handler(object):
    def __init__(self):
        """
        An object that handles operations with .h5 files
        """
        self.label_encoder = {}

    def label_encode(self, item: str) -> int:
        if item not in self.label_encoder:
            self.label_encoder[item] = len(self.label_encoder)
        encoded_item = self.label_encoder[item]
        return encoded_item

    def save_features_to_h5(
        self,
        layer_i_features_fname: str,
        indices: tuple,
        subset: tuple,
        label: tuple,
        hidden_states: torch.tensor,
        transformer_model_config,
    ):
        """
        A function that saves the indices of the processed examples,
        the subset labels (tr/va/te), and the layer features of the transformer model
        :param layer_i_features_fname: The name of .h5 file to store the layer features
        :param indices: A list of the batch indices
        :param subset: A list of the batch subset labels
        :param label: A list of the batch target labels
        :param hidden_states: A tensor of the batch hidden states
        :return:
        """
        batch = {
            "index": indices.numpy().astype(int),
            "subset": np.asarray(list(subset)).astype("S20"),
            "label": np.asarray(list(label)).astype("S20"),
            "hidden_states": hidden_states.astype(float),
        }

        if not os.path.isfile(layer_i_features_fname):
            with h5py.File(layer_i_features_fname, "w") as h5f:
                for field, feats in batch.items():
                    h5f.create_dataset(
                        field,
                        data=feats,
                        maxshape=(None, transformer_model_config.hidden_size)
                        if field == "hidden_states"
                        else (None,),
                        chunks=True,
                    )
        else:
            with h5py.File(layer_i_features_fname, "a") as h5f:
                for field, feats in batch.items():
                    feats_shape = feats.shape[0]
                    resize_shape = h5f[field].shape[0] + feats_shape

                    h5f[field].resize(
                        (resize_shape, transformer_model_config.hidden_size)
                    ) if field == "hidden_states" else h5f[field].resize(
                        (resize_shape,)
                    )
                    h5f[field][-feats_shape:] = feats

    def load_features_from_h5(
        self,
        probe_task: str,
        model_name: str,
        layer_num: int,
        feature_dir: str = "features",
    ) -> list:
        """
        A function that loads the .h5 file storing the features
        :param probe_task: The probe task name
        :param model_name: The transformer model name
        :param layer_num: An ordinal number of the transformer model layer
        :param feature_dir: The feature directory name
        :param dataset_arguments: The dataset config file, to be removed
        :return: A dict of arrays (example index, hidden state, target label) for each partition
        """
        features = {subset: [] for subset in ("tr", "va", "te")}
        layer_h5_fname = f"layer_{layer_num}.h5"
        model_name = model_name.replace("/", "-")
        feature_h5_filename_path = os.path.join(
            feature_dir, probe_task, model_name, layer_h5_fname
        )
        h5_features = h5py.File(feature_h5_filename_path, "r")
        convert_numpy_bites_to_str = lambda x: x.decode("utf-8")

        for index, subset, label, hidden_states in zip(
            h5_features[("index")],
            map(convert_numpy_bites_to_str, h5_features[("subset")]),
            map(convert_numpy_bites_to_str, h5_features[("label")]),
            h5_features[("hidden_states")],
        ):
            index_tensor = torch.tensor(index)
            hidden_states_tensor = torch.from_numpy(hidden_states).float()
            label_tensor = torch.tensor(self.label_encode(label))
            features[subset].append((index_tensor, hidden_states_tensor, label_tensor))
        os.remove(feature_h5_filename_path)
        return features


class Featurizer(object):
    def __init__(self, probe_task: str, model_name: str, args: ProbingArguments):
        """
        An object for pre-processing the probe task that creates features for each layer of the transformer model
        and saves them into .h5 files
        :param probe_task: The probe task name
        :param model_name: The transformer model name
        :param args: The probing arguments
        """
        self.args = args
        self.probe_task = probe_task
        self.model_name = model_name

        self.hdf5handler = HDF5Handler()
        self.mean_pooling = MeanMaskedPooling()
        self.probing_dataset = ProbingDataset(
            probe_task=self.probe_task,
            prepro_batch_size=self.args.prepro_batch_size,
            bucketing=self.args.bucketing,
        )
        self.transformer_model = TransformerModel(
            model_name=self.model_name, model_is_random=self.args.model_is_random
        )
        self.iterator = DataLoader(
            self.probing_dataset, batch_size=self.args.prepro_batch_size, shuffle=False
        )

    def convert_examples_to_features(self, feature_dir: str = "features"):
        probe_task_dir_path = create_saving_directory(
            probe_task=self.probe_task,
            model_name=self.model_name,
            save_dir_name=feature_dir,
        )

        for indices, subset, label, sentences, sentence_lengths in tqdm(
            self.iterator, desc="Converting examples to features..."
        ):
            encoded_batch = self.transformer_model.tokenizer(
                list(sentences),
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded_batch["input_ids"].to(self.args.device)
            attention_mask = encoded_batch["attention_mask"].to(self.args.device)
            with torch.no_grad():
                model_outputs = self.transformer_model.model(
                        input_ids, attention_mask, return_dict=True
                )
                model_outputs = (
                    model_outputs["hidden_states"]
                    if "hidden_states" in model_outputs
                    else model_outputs["encoder_hidden_states"]
                )
                for layer_i in range(self.transformer_model.config.num_hidden_layers):
                    layer_i = layer_i + 1
                    layer_i_features_fname = os.path.join(
                        probe_task_dir_path, f"layer_{layer_i}.h5"
                    )
                    hidden_states = (
                        self.mean_pooling.forward(
                            model_outputs[layer_i], attention_mask.bool()
                        )
                        .cpu()
                        .numpy()
                    )
                    self.hdf5handler.save_features_to_h5(
                        layer_i_features_fname,
                        indices,
                        subset,
                        label,
                        hidden_states,
                        self.transformer_model.config,
                    )
            gc.collect()
