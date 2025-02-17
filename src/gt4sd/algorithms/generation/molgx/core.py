"""MolGX Algorithm.

MolGX generation algorithm.
"""

import logging
from dataclasses import field
from typing import Any, ClassVar, Dict, Iterator, Optional, Tuple, TypeVar

from ....domains.materials import SmallMolecule, validate_molecules
from ....exceptions import InvalidItem
from ...core import AlgorithmConfiguration, GeneratorAlgorithm, Untargeted
from ...registry import ApplicationsRegistry
from .implementation import MolGXGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = type(None)
S = TypeVar("S", bound=SmallMolecule)


class MolGX(GeneratorAlgorithm[S, T]):
    """MolGX Algorithm."""

    def __init__(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T] = None,
    ):
        """Instantiate MolGX ready to generate items.

        Args:
            configuration: domain and application
                specification defining parameters, types and validations.
            target: a target for which to generate items.

        Example:
            An example for generating small molecules (SMILES) with given HOMO and LUMO energies:

                configuration = MolGXQM9Generator()
                molgx = MolGX(configuration=configuration, target=target)
                items = list(molgx.sample(10))
                print(items)
        """

        configuration = self.validate_configuration(configuration)
        # TODO there might also be a validation/check on the target input

        super().__init__(
            configuration=configuration,  # type:ignore
            target=target,  # type:ignore
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration[S, T],
        target: Optional[T],
    ) -> Untargeted:
        """Get the function to sample batches via the ConditionalGenerator.

        Args:
            configuration: helps to set up the application.
            target: context or condition for the generation. Unused in the algorithm.

        Returns:
            callable generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: MolGXGenerator = configuration.get_conditional_generator(  # type: ignore
            self.local_artifacts
        )
        return implementation.generate

    def validate_configuration(
        self, configuration: AlgorithmConfiguration[S, T]
    ) -> AlgorithmConfiguration[S, T]:
        # TODO raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration

    def sample(self, number_of_items: int = 100) -> Iterator[S]:
        """Generate a number of unique and valid items.

        Args:
            number_of_items: number of items to generate.
                Defaults to 100.

        Yields:
            the items.
        """
        if hasattr(self.configuration, "maximum_number_of_solutions"):
            maxiumum_number_of_molecules = int(
                getattr(self.configuration, "maximum_number_of_solutions")
            )
            if number_of_items > maxiumum_number_of_molecules:
                logger.warning(
                    f"current MolGX configuration can not support generation of {number_of_items} molecules..."
                )
                logger.warning(
                    f"to enable generation of {number_of_items} molecules, increase 'maximum_number_of_solutions' (currently set to {maxiumum_number_of_molecules})"
                )
                number_of_items = maxiumum_number_of_molecules
                logger.warning(f"generating at most: {maxiumum_number_of_molecules}...")
        return super().sample(number_of_items=number_of_items)


@ApplicationsRegistry.register_algorithm_application(MolGX)
class MolGXQM9Generator(AlgorithmConfiguration[SmallMolecule, Any]):
    """Configuration to generate compounds with given HOMO and LUMO energies."""

    algorithm_type: ClassVar[str] = "generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = "v0"

    homo_energy_value: Tuple[float, float] = field(
        default=(-0.5, 0.5),
        metadata=dict(description="Target HOMO energy range."),
    )
    lumo_energy_value: Tuple[float, float] = field(
        default=(-0.5, 0.5),
        metadata=dict(description="Target LUMO energy eange."),
    )
    number_of_candidates: int = field(
        default=10,
        metadata=dict(description="Number of feature vectors at a time."),
    )
    maximum_number_of_candidates: int = field(
        default=50,
        metadata=dict(
            description="Maximum number of candidate feature vector to consider."
        ),
    )
    maximum_number_of_molecules: int = field(
        default=100,
        metadata=dict(description="Maximum number of Molecules to obtain."),
    )
    maximum_number_of_solutions: int = field(
        default=10,
        metadata=dict(description="Maximum number of solutions to discover."),
    )
    maximum_number_of_nodes: int = field(
        default=200000,
        metadata=dict(
            description="Maximum number of search tree nodes in the graph exploration."
        ),
    )
    beam_size: int = field(
        default=1000,
        metadata=dict(description="Size of the beam during search."),
    )
    without_estimate: bool = field(
        default=True,
        metadata=dict(description="Generation without feature estimates."),
    )

    def get_target_description(self) -> Optional[Dict[str, str]]:
        """Get description of the target for generation.

        Returns:
            target description, returns None in case no target is used.
        """
        return None

    def get_conditional_generator(self, resources_path: str) -> MolGXGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.

        Returns:
            instance with :meth:`generate<gt4sd.algorithms.conditional_generation.molgx.implementation.MolGXGenerator.generate>` for generation.
        """
        return MolGXGenerator(
            resources_path=resources_path,
            homo_energy_value=self.homo_energy_value,
            lumo_energy_value=self.lumo_energy_value,
            number_of_candidates=self.number_of_candidates,
            maximum_number_of_candidates=self.maximum_number_of_candidates,
            maximum_number_of_molecules=self.maximum_number_of_molecules,
            maximum_number_of_solutions=self.maximum_number_of_solutions,
            maximum_number_of_nodes=self.maximum_number_of_nodes,
            beam_size=self.beam_size,
            without_estimate=self.without_estimate,
        )

    def validate_item(self, item: str) -> SmallMolecule:
        """Check that item is a valid SMILES.

        Args:
            item: a generated item that is possibly not valid.

        Raises:
            InvalidItem: in case the item can not be validated.

        Returns:
            the validated SMILES.
        """
        (
            molecules,
            _,
        ) = validate_molecules([item])
        if molecules[0] is None:
            raise InvalidItem(
                title="InvalidSMILES",
                detail=f'rdkit.Chem.MolFromSmiles returned None for "{item}"',
            )
        return SmallMolecule(item)
