from __future__ import annotations
from collections.abc import Callable


def name_for_index(
        name_prefix: str,
        maximum_index: int
) -> Callable[[int], str]:
    number_of_digits = len(f"{maximum_index}")
    def specific_function(specific_index: int) -> str:
        numeric_part = "{0:0{n}}".format(specific_index, n=number_of_digits)
        return f"{name_prefix}{numeric_part}"
    return specific_function
