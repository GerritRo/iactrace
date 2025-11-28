import jax
import equinox as eqx
from typing import Callable, Sequence, Union

from .paths import path_to_string, match_path


FilterFn = Callable[[tuple, any], bool]
PathSpec = Union[str, Sequence[str], FilterFn]


def make_filter(model, trainable: PathSpec = None, frozen: PathSpec = None):
    """
    Create a filter spec for eqx.partition.
    
    Args:
        model: The pytree to filter
        trainable: Paths to train (everything else frozen), can be:
            - str: single pattern
            - Sequence[str]: list of patterns
            - Callable[[path_tuple, leaf], bool]: custom filter
        frozen: Paths to freeze (everything else trainable), same types
    
    Returns:
        Filter spec (pytree of bools matching model structure)
    
    Patterns:
        'mirror_groups.rotations'       - exact match
        'mirror_groups.*'               - all direct children
        'mirror_groups.*.rotations'     - rotations in any child
        '**.rotations'                  - any path ending in rotations
    
    Examples:
        # Only train mirror rotations
        make_filter(tel, trainable='mirror_groups.*.rotations')
        
        # Train all mirror group params
        make_filter(tel, trainable='mirror_groups.**')
        
        # Freeze only sensor positions
        make_filter(tel, frozen='sensors.*.position')
        
        # Custom filter
        make_filter(tel, trainable=lambda path, leaf: 'rotation' in str(path))
    """
    if trainable is not None and frozen is not None:
        raise ValueError("Specify trainable OR frozen, not both")
    
    if trainable is None and frozen is None:
        frozen = []  # Default: everything trainable
    
    # Convert to callable
    if trainable is not None:
        filter_fn = _make_filter_fn(trainable, default=False)
    else:
        filter_fn = _make_filter_fn(frozen, default=True, invert=True)
    
    def leaf_filter(path, leaf):
        if not eqx.is_array(leaf):
            return False
        return filter_fn(path, leaf)
    
    return jax.tree_util.tree_map_with_path(
        leaf_filter,
        model,
        is_leaf=eqx.is_array
    )


def _make_filter_fn(spec: PathSpec, default: bool, invert: bool = False):
    """Convert PathSpec to filter function."""
    if callable(spec):
        fn = spec
    else:
        if isinstance(spec, str):
            patterns = [spec]
        else:
            patterns = list(spec)
        
        def fn(path, leaf):
            path_str = path_to_string(path)
            return any(match_path(path_str, p) for p in patterns)
    
    if invert:
        return lambda path, leaf: not fn(path, leaf)
    return fn


def partition(model, trainable: PathSpec = None, frozen: PathSpec = None):
    """
    Partition model into (trainable, frozen).
    
    Args:
        model: Equinox module
        trainable: What to train (else frozen)
        frozen: What to freeze (else trainable)
    
    Returns:
        (trainable_tree, frozen_tree) for use with eqx.combine
    """
    filter_spec = make_filter(model, trainable, frozen)
    return eqx.partition(model, filter_spec)


def trainable_params(model, spec: PathSpec):
    """Get only trainable parameters."""
    return partition(model, trainable=spec)[0]


def show_structure(model):
    """Print model structure with paths - useful for debugging filter patterns."""
    def print_leaf(path, leaf):
        path_str = path_to_string(path)
        if eqx.is_array(leaf):
            print(f"  {path_str}: {leaf.shape} {leaf.dtype}")
        return leaf
    
    print("Model structure:")
    jax.tree_util.tree_map_with_path(print_leaf, model, is_leaf=eqx.is_array)