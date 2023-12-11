from typing import Callable, Dict, List

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
