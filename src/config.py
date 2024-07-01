"""
Contains definitions relating to the configuration of how and what
types of training will be performed
"""

import hp_params
import tomllib


class ConfigData:
    """
    Read and process configurations from the `config.toml` file.
    These files are used to drive the training, selecting which type
    of training to do and how to do it.
    """

    def __init__(self, config_file_path: str):
        """
        Initialize the object.

        Parameters:
          config_file_path (str): String with the path to a TOML configuration file.

        Attributes:
          config_file (dict[str, Any]): The loaded TOML configuration file as a dictionary of options.

          num_qubits (int): The total number of qubits to use for training.

          ansatz (str): The type of ansatz to use.
          conv_layer (str): The name of the convolutional layer to use. Ignored if `ansatz != "conv"`.
          num_layers (int): The number of entangling layers to use. Ignored if `ansatz != "sel"`.

          dataset_size (int): The number of datapoints to use during training.
          batch_size (int): The number of subsamples to use during batching. If set to `0`, no batching is used.

          optimizer (str): The training library to use for optimization.
          max_iters (int): The maximum number of steps the optimizer is allowed to take.
          abstol (float): The error function value bellow which training stops.
          setp_size (float): Initial step size for the optimizing algorithm.

          x0 (float): Left domain boundary.
          xf (float): Right domain boundary.

          problem_type (str): The type of problem to solve.

          output_folder_name (str): The folder where training output will be generated.

          hp_params: (HagenPoiseuilleParams): The parameters for the Hagen-Poiseuille problem to be solved. Only used if `problem_type = "plane-hagen-poiseuille" or "plane-hagen-poiseuille"`
        """
        with open(config_file_path, "rb") as f:
            self.config_file = tomllib.load(f)

        self.num_qubits = self.config_file["computer"]["num_qubits"]

        self.ansatz = self.config_file["circuit"]["ansatz"]
        self.conv_layer = self.config_file["circuit"]["conv_layer"]
        self.num_layers = self.config_file["circuit"]["num_layers"]

        self.dataset_size = self.config_file["dataset"]["dataset_size"]
        self.batch_size = self.config_file["dataset"]["batch_size"]

        self.optimizer = self.config_file["training"]["optimizer"]
        self.max_iters = self.config_file["training"]["max_iters"]
        self.abstol = self.config_file["training"]["abstol"]
        self.step_size = self.config_file["training"]["step_size"]

        self.x0 = self.config_file["domain"]["x0"]
        self.xf = self.config_file["domain"]["xf"]

        self.problem_type = self.config_file["problem"]["problem_type"]

        self.output_folder_name = self.config_file["output"]["folder_name"]

        if self.problem_type == "hagen-poiseuille" or self.problem_type == "plane-hagen-poiseuille":
            self.hp_params = hp_params.HagenPoiseuilleParams(
                self.config_file["hagen-poiseuille-params"]["G"],
                self.config_file["hagen-poiseuille-params"]["R"],
                self.config_file["hagen-poiseuille-params"]["mu"]
            )
