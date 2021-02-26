from dataclasses import field, dataclass


@dataclass
class ProbingArguments(object):
    """
    An object to store the experiment arguments
    """

    seed: int = field(default=42, metadata={"help": "random seed for initialization"})
    prepro_batch_size: int = field(
        default=128, metadata={"help": "batch size for creating features"}
    )
    bucketing: bool = field(
        default=True,
        metadata={
            "help": "whether to perform char-level sequence bucketing for pre-processing"
        },
    )
    model_is_random: bool = field(
        default=False,
        metadata={"help": "whether to randomly initialize the transformer model"},
    )
    train_batch_size: int = field(
        default=256, metadata={"help": "batch size for training"}
    )
    eval_batch_size: int = field(
        default=128, metadata={"help": "batch size for evaluation"}
    )
    device: str = field(
        default="cuda", metadata={"help": "the device used during training"}
    )
    input_dim: int = field(default=768, metadata={"help": "input embedding shape"})
    num_hidden: int = field(
        default=250,
        metadata={"help": "number of hidden units in the non-linear classifier"},
    )
    max_iter: int = field(default=200, metadata={"help": "max number of epochs"})
    droupout_rate: float = field(
        default=0.2, metadata={"help": "dropout rate for the non-linear classifier"}
    )
    num_classes: int = field(default=2, metadata={"help": "number of target classes"})
    learning_rate: int = field(
        default=0.01, metadata={"help": "learning rate for the classifiers"}
    )
    clf: str = field(
        default="logreg",
        metadata={"help": "non-linear or linear classifier name (logreg, mlp)"},
    )
    num_kfold: int = field(
        default=5, metadata={"help": "number of folds for k-fold training"}
    )
    balanced: bool = field(
        default=True,
        metadata={
            "help": "whether to compute the weighted accuracy score if imbalanced"
        },
    )
