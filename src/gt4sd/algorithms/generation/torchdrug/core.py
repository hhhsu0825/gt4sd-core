"""Torchdrug generation algorithm."""

import logging
from typing import ClassVar, Optional, TypeVar

from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import GAFGenerator, GCPNGenerator, Generator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=str)


class TorchDrugGenerator(GeneratorAlgorithm[S, T]):
    def __init__(
        self, configuration: AlgorithmConfiguration, target: Optional[T] = None
    ):
        """TorchDrug generation algorithm.

        Args:
            configuration: domain and application specification, defining types
                and validations.  Currently supported algorithm versions are:
                "zinc250k_v0", "qed_v0" and "plogp_v0".
            target: unused since it is not a conditional generator.

        Example:
            An example for using a generative algorithm from TorchDrug:

                configuration = TorchDrugGCPN(algorithm_version="qed_v0")
                algorithm = TorchDrugGenerator(configuration=configuration)
                items = list(algorithm.sample(1))
                print(items)
        """

        configuration = self.validate_configuration(configuration)
        super().__init__(
            configuration=configuration,
            target=None,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Untargeted:
        """Get the function to sample batches.

        Args:
            configuration: helps to set up the application.
            target: context or condition for the generation. Unused in the algorithm.

        Returns:
            callable generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: Generator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.sample

    def validate_configuration(
        self, configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(TorchDrugGenerator)
class TorchDrugGCPN(AlgorithmConfiguration[str, None]):
    """
    Interface for TorchDrug Graph-convolutional policy network (GCPN) algorithm.
    Currently supported algorithm versions are "zinc250k_v0", "qed_v0" and "plogp_v0".
    """

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "zinc250k_v0"

    def get_conditional_generator(self, resources_path: str) -> GCPNGenerator:
        """Instantiate the actual generator implementation.
        Args:
            resources_path: local path to model files.
        Returns:
            instance with :meth:`sample<gt4sd.algorithms.generation.torchdrug.implementation.GCPNGenerator.sample>` method for generation.
        """
        self.generator = GCPNGenerator(resources_path=resources_path)
        return self.generator


@ApplicationsRegistry.register_algorithm_application(TorchDrugGenerator)
class TorchDrugGraphAF(AlgorithmConfiguration[str, None]):
    """
    Interface for TorchDrug flow-based autoregressive graph algorithm (GraphAF).
    Currently supported algorithm versions are "zinc250k_v0", "qed_v0" and "plogp_v0".
    """

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "zinc250k_v0"

    def get_conditional_generator(self, resources_path: str) -> GAFGenerator:
        """Instantiate the actual generator implementation.
        Args:
            resources_path: local path to model files.
        Returns:
            instance with :meth:`samples<gt4sd.algorithms.generation.torchdrug.implementation.GAFGenerator.sample>` method for generation.
        """
        self.generator = GAFGenerator(resources_path=resources_path)
        return self.generator
