from typing import Callable, Dict, List
from dimod import SampleSet


def name_for_index(
        name_prefix: str,
        maximum_index: int
    ) -> Callable[[int], str]:
    number_of_digits = len(f"{maximum_index}")
    def specific_function(specific_index: int) -> str:
        numeric_part = "{0:0{n}}".format(specific_index, n=number_of_digits)
        return f"{name_prefix}{numeric_part}"
    return specific_function


def spin_to_zero_or_one(spin_value: int) -> str:
    return "0" if spin_value > 0 else "1"


def as_bitstring(
        *,
        spin_variable_names: List[str],
        spin_mapping: Dict[str, int]
    ) -> str:
    return "".join(
        [spin_to_zero_or_one(spin_mapping[n]) for n in spin_variable_names]
    )


def bitstrings_to_energies(
        *,
        binary_variable_names: List[str],
        sample_set: SampleSet
    ) -> Dict[str, float]:
    return {
        as_bitstring(
                spin_variable_names=binary_variable_names,
                spin_mapping=s
            ): e
            for s, e in [(d.sample, d.energy) for d in sample_set.data()]
    }


def print_bitstrings(title_message: str, sample_set: SampleSet):
    print(title_message)
    print(
        "[v for v in sample_set.variables] = \n",
        [v for v in sample_set.variables]
    )
    print(
        "bitstrings in above variable order to energies =\n",
        bitstrings_to_energies(
            binary_variable_names=sample_set.variables,
            sample_set=sample_set
        )
    )
