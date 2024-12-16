import typing as tp
import math

import torch
from torch import Tensor
import torch.utils.data

Device = tp.Union[torch.device, str, int, None]
DType = tp.Optional[torch.dtype]


class SpeciesEnergies(tp.NamedTuple):
    r"""Tuple used in output of ANI models"""

    species: Tensor
    energies: Tensor


__all__ = [
    "ChemicalSymbolsToInts",
    "linspace",
    "cumsum_from_zero",
    "map_to_central",
    "fast_masked_select",
]


def linspace(start: float, stop: float, steps: int) -> tp.Tuple[float, ...]:
    r"""Pure python linspace

    Used to ensure repro of constants in case `numpy` changes its internal
    implementation.
    """
    return tuple(start + ((stop - start) / steps) * j for j in range(steps))


def cumsum_from_zero(input_: Tensor) -> Tensor:
    r"""Cumulative sum just like `torch.cumsum`, but result starts from 0"""
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum


def nonzero_in_chunks(tensor: Tensor, chunk_size: int = 2**31 - 1):
    r"""Flatten a tensor and applies nonzero in chunks of a given size

    Workaround for a limitation in PyTorch's nonzero function, which fails with a
    `RuntimeError` when applied to tensors with more than ``INT_MAX`` elements.

    The issue is documented in
    `PyTorch's repo <https://github.com/pytorch/pytorch/issues/51871>`_.
    """
    tensor = tensor.view(-1)
    num_splits = math.ceil(tensor.numel() / chunk_size)

    if num_splits <= 1:
        return tensor.nonzero().view(-1)

    # Split tensor into chunks, and for each chunk find nonzero elements and
    # adjust the indices in each chunk to account for their original position
    # in the tensor. Finally collect the results
    offset = 0
    nonzero_chunks: tp.List[Tensor] = []
    for chunk in torch.chunk(tensor, num_splits):
        nonzero_chunks.append(chunk.nonzero() + offset)
        offset += chunk.shape[0]
    return torch.cat(nonzero_chunks).view(-1)


def fast_masked_select(x: Tensor, mask: Tensor, idx: int) -> Tensor:
    r"""Has the same effect as `torch.masked_select` but faster"""
    # x.index_select(0, tensor.view(-1).nonzero().view(-1)) is EQUIVALENT to:
    # torch.masked_select(x, tensor) but FASTER
    # nonzero_in_chunks calls tensor.view(-1).nonzero().view(-1)
    # but support very large tensors, with numel > INT_MAX
    return x.index_select(idx, nonzero_in_chunks(mask))


def map_to_central(coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
    r"""Map atoms outside the unit cell into the cell using PBC

    Args:
        coordinates: |coords|
        cell: |cell|
        pbc: |pbc|
    Returns:
        Tensor of coordinates of atoms mapped to the unit cell.
    """
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    inv_cell = torch.inverse(cell)
    coordinates_cell = torch.matmul(coordinates, inv_cell)
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor() * pbc
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return torch.matmul(coordinates_cell, cell)


class ChemicalSymbolsToInts(torch.nn.Module):
    r"""Helper that can be called to convert chemical symbol string to integers

    On initialization the class should be supplied with a `list` of `str`. The returned
    instance is a callable object, which can be called with an arbitrary list of the
    supported species that is converted into an integer tensor. Usage
    example:

    .. code-block:: python

        from torchani.utils import ChemicalSymbolsToInts
        # We initialize ChemicalSymbolsToInts with the supported species
        elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        species_to_tensor = ChemicalSymbolsToInts(elements)
        species_convert = ['C', 'S', 'O', 'F', 'H', 'H']
        # We have a species list which we want to convert to an index tensor
        index_tensor = species_to_tensor(species_convert)
        # index_tensor is now [1, 4, 3, 5, 0, 0]

    Args:
        symbols: |symbols|
    """

    def __init__(self, symbols: tp.Sequence[str], device: Device = None):
        super().__init__()
        if isinstance(symbols, str):
            raise ValueError("symbols must be a sequence of str, but it can't be a str")
        int_dict = {s: i for i, s in enumerate(symbols)}
        self.symbol_dict = int_dict
        self.register_buffer("_dummy", torch.empty(0, device=device), persistent=False)

    def forward(self, species: tp.List[str]) -> Tensor:
        r"""Converts a list of chemical symbols to an integer tensor"""
        # This can't be an in-place loop to be jit-compilable
        numbers_list: tp.List[int] = []
        for x in species:
            numbers_list.append(self.symbol_dict[x])
        return torch.tensor(numbers_list, dtype=torch.long, device=self._dummy.device)

    def __len__(self) -> int:
        return len(self.symbol_dict)


# Useful function for simple classes meant as user extension points
def _validate_user_kwargs(
    clsname: str,
    names_dict: tp.Dict[str, tp.Sequence[str]],
    kwargs: tp.Dict[str, tp.Union[tp.Tuple, tp.List, float]],
    trainable: tp.Sequence[str],
) -> None:
    _num_tensors = sum(len(seq) for seq in names_dict.values())
    kwargs_set: tp.Set[str] = set()
    for v in names_dict.values():
        kwargs_set = kwargs_set.union(v)

    if len(kwargs_set) != _num_tensors:
        raise ValueError("tensor names must be unique")

    if set(kwargs) != kwargs_set:
        raise ValueError(
            f"Expected arguments '{', '.join(kwargs_set)}'"
            f" but got '{', '.join(kwargs.keys())}'"
            f" Maybe you forgot '*_tensors = [..., 'argname']'"
            f" when defining the class?"
        )

    for names, tensors in names_dict.items():
        _seqs = [
            v for k, v in kwargs.items() if k in names and isinstance(v, (tuple, list))
        ]
        if _seqs and not all(len(s) == len(_seqs[0]) for s in _seqs):
            raise ValueError(
                f"Tuples or lists passed to {clsname}"
                " corresponding to '{tensors}' must have the same len"
            )

    if not set(trainable).issubset(kwargs_set):
        raise ValueError(
            f"trainable={trainable} could not be found in {kwargs_set}"
        )
