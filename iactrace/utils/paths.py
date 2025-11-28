import jax

def path_to_string(path_tuple):
    """Convert JAX path tuple to dot-separated string."""
    parts = []
    for p in path_tuple:
        if isinstance(p, jax.tree_util.SequenceKey):
            parts.append(f'[{p.idx}]')
        elif isinstance(p, jax.tree_util.DictKey):
            parts.append(str(p.key))
        elif isinstance(p, jax.tree_util.GetAttrKey):
            parts.append(str(p.name))
        else:
            parts.append(str(p))
    return '.'.join(parts)


def match_path(path_str, pattern):
    """
    Match path string against pattern.
    
    Patterns:
        'mirror_groups.rotations'       - exact match
        'mirror_groups.*'               - all leaves under mirror_groups  
        'mirror_groups.*.rotations'     - rotations in any mirror group
        '**.rotations'                  - any path ending in rotations
    """
    path_parts = path_str.split('.')
    pattern_parts = pattern.split('.')
    
    return match_parts(path_parts, pattern_parts)


def match_parts(path_parts, pattern_parts):
    """Recursive glob-style matching."""
    if not pattern_parts:
        return not path_parts
    if not path_parts:
        return pattern_parts == ['**'] or all(p == '*' for p in pattern_parts)
    
    p = pattern_parts[0]
    
    if p == '**':
        # Match zero or more path components
        if len(pattern_parts) == 1:
            return True
        for i in range(len(path_parts) + 1):
            if match_parts(path_parts[i:], pattern_parts[1:]):
                return True
        return False
    elif p == '*':
        # Match exactly one component
        return match_parts(path_parts[1:], pattern_parts[1:])
    else:
        # Exact match
        return path_parts[0] == p and _match_parts(path_parts[1:], pattern_parts[1:])
