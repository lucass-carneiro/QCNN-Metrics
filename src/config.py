import hp_params
import tomllib


class ConfigData:
    def __init__(self, config_file_path: str):
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

        if self.problem_type == "hagen-poiseuille":
            self.hp_params = hp_params.HagenPoiseuilleParams(
                self.config_file["hagen-poiseuille-params"]["G"],
                self.config_file["hagen-poiseuille-params"]["R"],
                self.config_file["hagen-poiseuille-params"]["mu"]
            )
