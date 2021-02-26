import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, BartModel, XLMRobertaTokenizer


class TransformerModel(object):
    def __init__(self, model_name: str, model_is_random: bool = False):
        """
        A wrapper for the transformer model
        :param model_name: The name of the transformer model (HuggingFace)
        :param model_is_random: Whether to randomly initialize the weights of the model
        """
        self.config = self._load_config(model_name)
        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)

        if model_is_random:
            self.model.init_weights()

        if torch.cuda.is_available():
            self.model.cuda()

    def _load_config(self, model_name: str):
        return AutoConfig.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )

    def _load_model(self, model_name: str):
        if model_name in ("facebook/mbart-large-cc25"):
            return BartModel.from_pretrained(model_name, config=self.config).eval()
        return AutoModel.from_pretrained(model_name, config=self.config).eval()

    def _load_tokenizer(self, model_name: str):
        if model_name in ("microsoft/Multilingual-MiniLM-L12-H384"):
            return XLMRobertaTokenizer.from_pretrained(model_name, config=self.config)
        return AutoTokenizer.from_pretrained(model_name, config=self.config)


class MeanMaskedPooling(torch.nn.Module):
    def __init__(self):
        """
        An object to perform Mean Pooling that ignores PAD-token representations
        """
        super(MeanMaskedPooling, self).__init__()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor):
        lengths = pad_mask.sum(1).float()
        x = x.masked_fill((~pad_mask).unsqueeze(-1), 0.0)
        scaling = x.size(1) / lengths
        scaling = scaling.masked_fill(lengths == 0, 1.0).unsqueeze(-1)
        x = x.mean(1) * scaling
        return x


class Linear(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        """
        pytorch Logistic Regression
        :param input_dim: The shape of the transformer model representations
        :param num_classes: The number of the probe task classes
        """
        super(Linear, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        return x


class NonLinear(torch.nn.Module):
    def __init__(
        self, input_dim: int, num_hidden: int, num_classes: int, dropout_rate: float
    ):
        """
        pytorch Multi-layer Perceptron
        :param input_dim: The shape of the transformer model representations
        :param num_hidden: The number of hidden units
        :param num_classes: The number of the probe task classes
        :param dropout_rate: The dropout rate
        """
        super(NonLinear, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.fc1 = torch.nn.Linear(self.input_dim, self.num_hidden)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.activation = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(self.num_hidden, self.num_classes)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
