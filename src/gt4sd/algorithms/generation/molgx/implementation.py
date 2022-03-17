"""Implementation of MolGX conditional generators."""

import logging
import os
from typing import Any, Dict, List, Tuple

from ....extras import EXTRAS_ENABLED

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if EXTRAS_ENABLED:
    from ....extras.molgx import *  # Hsu

    class MolGXGenerator:
        """Interface for MolGX generator."""

        def __init__(
            self,
            resources_path: str,
            model_name: str,
            homo_energy_value: Tuple[float, float] = (-0.5, 0.5),
            lumo_energy_value: Tuple[float, float] = (-0.5, 0.5),
            number_of_candidates: int = 10,
            maximum_number_of_candidates: int = 50,
            maximum_number_of_molecules: int = 100,
            maximum_number_of_solutions: int = 10,
            maximum_number_of_nodes: int = 200000,
            beam_size: int = 1000,
            without_estimate: bool = True,
        ) -> None:
            """Instantiate a MolGX generator.

            Args:
                resources_path: path to the resources for model loading.
                homo_energy_value: target HOMO energy range. Defaults to (-0.5, 0.5).
                lumo_energy_value: target LUMO energy range. Defaults to (-0.5, 0.5).
                number_of_candidates: number of feature vectors at a time. Defaults to 10.
                maximum_number_of_candidates: maximum number of candidate feature vector to consider. Defaults to 50.
                maximum_number_of_molecules: maximum number of molecules to obtain. Defaults to 100.
                maximum_number_of_solutions: maximum number of solutions to discover. Defaults to 10.
                maximum_number_of_nodes: maximum number of search tree nodes in the graph exploration. Defaults to 200000.
                beam_size: size of the beam during search. Defaults to 1000.
                without_estimate: generation without feature estimates. Defaults to False.

            Raises:
                RuntimeError: in the case extras are disabled.
            """
            if not EXTRAS_ENABLED:
                raise RuntimeError("Can't instantiate MolGXGenerator, extras disabled!")

            # loading artifacts
            self.resources_path = resources_path
            self.model_name = model_name
            self.amd = self.load_molgx(self.resources_path)
            self.molecules_data, self.target_property = self.amd.LoadPickle(self.model_name)
            # algorithm parameters
            self._homo_energy_value = homo_energy_value
            self._lumo_energy_value = lumo_energy_value
            self._number_of_candidates = number_of_candidates
            self._maximum_number_of_candidates = maximum_number_of_candidates
            self._maximum_number_of_molecules = maximum_number_of_molecules
            self._maximum_number_of_solutions = maximum_number_of_solutions
            self._maximum_number_of_nodes = maximum_number_of_nodes
            self._beam_size = beam_size
            self._without_estimate = without_estimate
            self._parameters = self._create_parameters_dictionary()

        @staticmethod
        def load_molgx(resource_path: str) -> molgxsdk:
            """Load MolGX model.

            Args:
                resource_path: path to the resources for model loading.
                tag_name: tag for the pretrained model.

            Returns:
                MolGX model SDK.
            """

            return MolgxSdk(
                dir_pickle=resource_path,
            )

        def _create_parameters_dictionary(self) -> Dict[str, Any]:
            """Create parameters dictionary.

            Returns:
                the parameters to run MolGX.
            """
            self.target_property["homo"] = self.homo_energy_value
            self.target_property["lumo"] = self.lumo_energy_value
            parameters: Dict[str, Any] = {}
            parameters["target_property"] = self.target_property
            parameters["num_candidate"] = self.number_of_candidates
            parameters["max_candidate"] = self.maximum_number_of_candidates
            parameters["max_molecule"] = self.maximum_number_of_molecules
            parameters["max_solution"] = self.maximum_number_of_solutions
            parameters["max_node"] = self.maximum_number_of_nodes
            parameters["beam_size"] = self.beam_size
            parameters["without_estimate"] = self.without_estimate
            return parameters

        @property
        def homo_energy_value(self) -> Tuple:
            return self._homo_energy_value

        @homo_energy_value.setter
        def homo_energy_value(self, value: Tuple) -> None:
            self._homo_energy_value = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def lumo_energy_value(self) -> Tuple:
            return self._lumo_energy_value

        @lumo_energy_value.setter
        def lumo_energy_value(self, value: Tuple) -> None:
            self._lumo_energy_value = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def number_of_candidates(self) -> int:
            return self._number_of_candidates

        @number_of_candidates.setter
        def number_of_candidates(self, value: int) -> None:
            self._number_of_candidates = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_candidates(self) -> int:
            return self._maximum_number_of_candidates

        @maximum_number_of_candidates.setter
        def maximum_number_of_candidates(self, value: int) -> None:
            self._maximum_number_of_candidates = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_molecules(self) -> int:
            return self._maximum_number_of_molecules

        @maximum_number_of_molecules.setter
        def maximum_number_of_molecules(self, value: int) -> None:
            self._maximum_number_of_molecules = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_solutions(self) -> int:
            return self._maximum_number_of_solutions

        @maximum_number_of_solutions.setter
        def maximum_number_of_solutions(self, value: int) -> None:
            self._maximum_number_of_solutions = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def maximum_number_of_nodes(self) -> int:
            return self._maximum_number_of_nodes

        @maximum_number_of_nodes.setter
        def maximum_number_of_nodes(self, value: int) -> None:
            self._maximum_number_of_nodes = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def beam_size(self) -> int:
            return self._beam_size

        @beam_size.setter
        def beam_size(self, value: int) -> None:
            self._beam_size = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def without_estimate(self) -> bool:
            return self._without_estimate

        @without_estimate.setter
        def without_estimate(self, value: bool) -> None:
            self._without_estimate = value
            self.parameters = self._create_parameters_dictionary()

        @property
        def parameters(self) -> Dict[str, Any]:
            return self._parameters

        @parameters.setter
        def parameters(self, value: Dict[str, Any]) -> None:
            parameters = self._create_parameters_dictionary()
            parameters.update(value)
            self._parameters = parameters

        def generate(self) -> List[str]:
            """Sample random molecules.

            Returns:
                sampled molecule (SMILES).
            """
            # generate molecules
            logger.info(
                f"running MolGX with the following parameters: {self.parameters}"
            )
            molecules_df = self.amd.GenMols(self.molecules_data, self.parameters)
            logger.info("MolGX run completed")
            return molecules_df["SMILES"].tolist()

else:
    logger.warning("install molgx extras to use MolGX")
