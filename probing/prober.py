import gc
import copy
import torch

import numpy as np
from typing import Optional
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, Dataset

from probing.arguments import ProbingArguments
from probing.modeling_utils import Linear, NonLinear
from probing.utils import FeatureDataset, HDF5Handler, init_seed, save_results


class Prober(object):
    def __init__(
        self,
        args: ProbingArguments,
        probe_task: str,
        model_name: str,
        layer_num: Optional[str] = None,
        features: Optional[dict] = None,
    ):
        """
        An object that performs tuning the L2 parameter, training the classifier, and probing the layer
        :param args: The experiment arguments
        :param probe_task: The probe task name
        :param model_name: [Optional] The name of the transformer model
        :param layer_num: [Optional] The number of the layer to probe
        :param features: [Optional] Features for each partition
        """
        self.args = args
        self.probe_task = probe_task
        self.model_name = model_name
        self.layer_num = layer_num
        self.hdf5handler = HDF5Handler()
        self.features = self._get_features(features)
        self.train_dataset = self._load_dataset(self.features, "tr")
        self.eval_dataset = self._load_dataset(self.features, "va")
        self.test_dataset = self._load_dataset(self.features, "te")

    def _load_dataset(self, _features: dict, subset: str):
        if subset not in _features:
            return None
        return FeatureDataset(_features[subset])

    def _get_features(self, features: Optional[dict] = None):
        if features is None:
            return self.hdf5handler.load_features_from_h5(
                probe_task=self.probe_task,
                model_name=self.model_name,
                layer_num=self.layer_num,
            )
        return features

    def _get_train_sampler(self):
        return RandomSampler(self.train_dataset)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Prober: training requires a train_dataset.")

        self.args.input_dim = self.train_dataset[0][1].size(0)
        train_sampler = self._get_train_sampler()

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset):
        return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset):
        if eval_dataset is None:
            raise ValueError("Prober: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

    def get_test_dataloader(self, test_dataset: Dataset):
        test_sampler = self._get_eval_sampler(test_dataset)

        return DataLoader(
            test_dataset, sampler=test_sampler, batch_size=self.args.eval_batch_size
        )

    def create_clf_optimizer_criterion(
        self, l2reg: float, label_weight: Optional[dict] = None
    ):
        if label_weight:
            label_weight = torch.FloatTensor(list(label_weight.values()))
        clf = (
            Linear(self.args.input_dim, self.args.num_classes)
            if self.args.clf == "logreg"
            else NonLinear(
                self.args.input_dim,
                self.args.num_hidden,
                self.args.num_classes,
                self.args.droupout_rate,
            )
        ).to(self.args.device)
        criterion = (
            torch.nn.CrossEntropyLoss()
            if label_weight is None
            else torch.nn.CrossEntropyLoss(weight=label_weight)
        )
        criterion.to(self.args.device)
        optimizer = torch.optim.Adam(clf.parameters())
        optimizer.param_groups[0]["weight_decay"] = l2reg
        return clf, criterion, optimizer

    def train_epoch(self, train_dataloader: DataLoader):
        epoch_tr_losses = []
        self.clf.train()
        for i, x, y in train_dataloader:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            self.optimizer.zero_grad()
            prediction = self.clf(x)
            # loss
            loss = self.criterion(prediction, y)
            epoch_tr_losses.append(loss.item())
            # backward
            loss.backward()
            # update params
            self.optimizer.step()
        epoch_tr_loss = np.mean(epoch_tr_losses)
        return epoch_tr_loss

    def compute_weights(self, dataset: FeatureDataset):
        target_labels = [item[-1].item() for item in dataset.features]
        unique_labels = np.unique(target_labels)
        label_weights = compute_class_weight("balanced", unique_labels, target_labels)
        label_weights = dict(zip(unique_labels, label_weights))
        return label_weights

    def evaluate(
        self,
        eval_dataloader: DataLoader,
        label_weight: Optional[dict] = None,
        bootstrap: Optional[bool] = False,
    ):
        te_losses = []
        te_scores = []
        self.clf.eval()
        weighted_accuracy_score = lambda x, y: accuracy_score(y, x, label_weight)
        with torch.no_grad():
            for i, x, y in eval_dataloader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                clf_output = self.clf(x)
                loss = self.criterion(clf_output, y)
                te_losses.append(loss.item())
                _, prediction = clf_output.data.max(1)
                if self.args.balanced:
                    batch_scores = prediction.eq(y).float().tolist()
                else:
                    batch_scores = list(
                        map(
                            weighted_accuracy_score,
                            prediction.cpu().unsqueeze(1),
                            y.cpu().unsqueeze(1),
                        )
                    )
                te_scores.extend(batch_scores)
        te_score = round(np.mean(te_scores), 3)
        epoch_te_loss = round(np.mean(te_losses), 3)
        if bootstrap:
            save_results(
                self.probe_task,
                te_scores,
                self.model_name,
                "_".join([self.args.clf, "layer", str(self.layer_num)]),
            )
        gc.collect()
        return te_score, epoch_te_loss

    def fit(self):
        l2regs = [10 ** t for t in range(-5, 0)]
        tr_label_weights = (
            None if self.args.balanced else self.compute_weights(self.train_dataset)
        )
        va_label_weights = (
            None if self.args.balanced else self.compute_weights(self.eval_dataset)
        )

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        grid_params = {"va_score": -1, "l2reg": l2regs[0], "model": None}
        # grid-search
        for l2reg in l2regs:
            # k-fold training
            print(f"Training {self.args.clf} with param L2 {l2reg}...")
            for k in range(self.args.num_kfold):
                num_epoch = 0
                kfold_score = 0
                best_va_loss = 10
                kfold_tr_losses = []
                kfold_va_losses = []
                stop_training = False
                # re-sample train data
                train_dataloader = self.get_train_dataloader()
                init_seed(self.args.seed)
                (
                    self.clf,
                    self.criterion,
                    self.optimizer,
                ) = self.create_clf_optimizer_criterion(l2reg, tr_label_weights)
                # train
                while not stop_training and num_epoch <= self.args.max_iter:
                    num_epoch += 1
                    epoch_tr_loss = self.train_epoch(train_dataloader)
                    kfold_tr_losses.append(epoch_tr_loss)
                    # evaluate
                    epoch_score, epoch_va_loss = self.evaluate(
                        eval_dataloader, va_label_weights
                    )
                    kfold_va_losses.append(epoch_va_loss)
                    # update accuracy
                    if kfold_score < epoch_score:
                        kfold_score = epoch_score
                    # early stopping
                    if epoch_va_loss < best_va_loss:
                        best_va_loss = epoch_va_loss
                    else:
                        stop_training = True
                # update best params
                if kfold_score > grid_params["va_score"]:
                    grid_params["va_score"] = kfold_score
                    grid_params["l2reg"] = l2reg
                    grid_params["model"] = copy.deepcopy(self.clf)
                    grid_params["tr_loss"] = round(np.mean(kfold_tr_losses), 3)
                    grid_params["va_loss"] = round(np.mean(kfold_va_losses), 3)
                gc.collect()
        return grid_params

    def probe(self):
        grid_params = self.fit()
        self.clf = grid_params["model"]
        print(f"Best L2: {grid_params['l2reg']}")
        print(f"Train Loss: {grid_params['tr_loss']}")
        print(f"Validation Loss: {grid_params['va_loss']}")
        print(f"Validation Score: {grid_params['va_score']}")
        test_dataloader = self.get_eval_dataloader(self.test_dataset)
        te_label_weights = (
            None if self.args.balanced else self.compute_weights(self.test_dataset)
        )
        te_score, te_loss = self.evaluate(test_dataloader, te_label_weights, True)
        print(f"Test Loss: {te_loss}")
        print(f"Test Score: {te_score}")
        print("*" * 30)
        print()
        del grid_params["model"]
        results = copy.deepcopy(grid_params)
        results.update(
            {
                "label_encoder": self.hdf5handler.label_encoder,
                "te_loss": te_loss,
                "te_score": te_score,
            }
        )
        return results
