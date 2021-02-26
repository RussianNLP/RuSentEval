import warnings

warnings.simplefilter(action="ignore", category=(FutureWarning, DeprecationWarning))

from probing.prober import Prober
from probing.arguments import ProbingArguments
from probing.utils import Featurizer, save_results


class Experiment(object):
    def __init__(
        self,
        probe_tasks: list,
        model_name: str,
        args: ProbingArguments,
        result_dir: str = "results",
    ):
        """
        An object to perform the experiment for the probing tasks
        :param probe_tasks: A list of probing tasks to probe
        :param model_name: The transformer model name
        :param args: The experiment arguments
        """
        self.probe_tasks = probe_tasks
        self.model_name = model_name
        self.args = args
        self.result_dir = result_dir

    def run(self):
        print(f"Start probing for the following tasks: {self.probe_tasks}")

        for probe_task in self.probe_tasks:
            probe_task_results = {}
            probe_task_results["model_name"] = self.model_name
            probe_task_results["probe_task"] = probe_task
            print(f"Probing the {probe_task} task...")

            probe_featurizer = Featurizer(probe_task, self.model_name, self.args)
            probe_featurizer.convert_examples_to_features()

            for layer_num in range(
                probe_featurizer.transformer_model.config.num_hidden_layers
            ):
                layer_num = layer_num + 1
                print(f"Layer: {layer_num}...")
                prober = Prober(self.args, probe_task, self.model_name, layer_num)
                prober.args.num_classes = len(prober.hdf5handler.label_encoder)
                probe_layer_results = prober.probe()
                probe_task_results[layer_num] = probe_layer_results
            probe_task_results["probing_arguments"] = prober.args.__dict__

            save_results(
                probe_task,
                probe_task_results,
                self.model_name,
                self.args.clf,
                self.result_dir,
            )
