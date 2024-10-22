"""
Contains handlers for ouputting and recovering ADIOS2 training data.
"""

import adios2

import os
import shutil

import logging
logger = logging.getLogger(__name__)


class Output:
    """
    Holds a stream-writtable ADIOS2 file and creates new output files with the
    the correct naming convention for checkpointing.
    """

    def __init__(self, output_name: str, config_file: str):
        """
        Initializes the object

        Parameters:
          output_name (str): The output folder name where data will be stored.
          config_file (str): The path of the configuration file, which will be copyied to the output data folder.
        """

        self.output_name = output_name
        self.config_file = os.path.join(
            self.output_name, os.path.basename(config_file))

        # Create output dir
        if not os.path.exists(self.output_name):
            os.makedirs(self.output_name)

        # Copy config
        if not os.path.exists(self.config_file):
            shutil.copy2(config_file, self.config_file)

        # Find all .bp files and extract last checkpoint from file name
        file_list = map(os.path.splitext, os.listdir(self.output_name))
        file_list = filter(lambda x: x[1] == ".bp", file_list)
        file_list = list(file_list)

        # Recover data from previous checkpoint
        if len(file_list) != 0:
            self.last_checkpoint = int(file_list[-1][0])
            logger.info(
                f"Recovering data from checkpoint {self.last_checkpoint}"
            )

            prev_output_file = os.path.join(
                self.output_name,
                f"{self.last_checkpoint}.bp"
            )

            with adios2.FileReader(prev_output_file) as s:
                attrs = s.available_attributes()
                vars = s.available_variables()

                steps = int(vars["weights"]["AvailableStepsCount"])

                self.first_iter = int(attrs["last_iter"]["Value"])
                self.weights = s.read("weights", step_selection=[steps - 1, 1])
                self.recovered = True
        else:
            self.last_checkpoint = -1
            self.first_iter = None
            self.weights = None
            self.recovered = False

        self.output_file = os.path.join(
            self.output_name,
            f"{self.last_checkpoint + 1}.bp"
        )

        # Create ADIOS2 stream
        self.output_stream = adios2.Stream(self.output_file, "w")

    def __del__(self):
        """
        Deinitializes the object and closes the ADIOS2 output stream.
        """
        logger.info("Closing output stream")
        self.output_stream.close()
