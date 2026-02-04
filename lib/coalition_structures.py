"""
Generate all possible coalition structures for n players.

A coalition structure is a partition of the set of players into disjoint coalitions.
For n players, the number of possible coalition structures is the Bell number B(n).

Example:
    n=3: B(3) = 5 structures
    n=4: B(4) = 15 structures
    n=5: B(5) = 52 structures
"""

from typing import List, Set, Tuple
from itertools import combinations


def generate_set_partitions(elements: List[str]) -> List[List[Set[str]]]:
    """
    Generate all set partitions of a list of elements.

    Args:
        elements: List of element names (e.g., ['W', 'T', 'C'])

    Returns:
        List of partitions, where each partition is a list of sets.

    Example:
        >>> generate_set_partitions(['W', 'T'])
        [[{'W'}, {'T'}], [{'W', 'T'}]]

        >>> generate_set_partitions(['W', 'T', 'C'])
        [[{'W'}, {'T'}, {'C'}],
         [{'W', 'T'}, {'C'}],
         [{'W', 'C'}, {'T'}],
         [{'T', 'C'}, {'W'}],
         [{'W', 'T', 'C'}]]
    """
    if len(elements) == 0:
        return [[]]

    if len(elements) == 1:
        return [[{elements[0]}]]

    # Take the first element
    first = elements[0]
    rest = elements[1:]

    # Get all partitions of the remaining elements
    rest_partitions = generate_set_partitions(rest)

    result = []

    for partition in rest_partitions:
        # Option 1: Add first element as its own singleton coalition
        result.append([{first}] + partition)

        # Option 2: Add first element to each existing coalition
        for i in range(len(partition)):
            new_partition = [coalition.copy() for coalition in partition]
            new_partition[i].add(first)
            result.append(new_partition)

    return result


def partition_to_string(partition: List[Set[str]]) -> str:
    """
    Convert a partition to a readable string format.

    Args:
        partition: List of sets representing coalitions

    Returns:
        String representation, e.g., '(WTC)' or '( )' or '(WT)(CF)'

    Examples:
        >>> partition_to_string([{'W'}, {'T'}, {'C'}])
        '( )'
        >>> partition_to_string([{'W', 'T', 'C'}])
        '(WTC)'
        >>> partition_to_string([{'W', 'T'}, {'C'}])
        '(WT)'
        >>> partition_to_string([{'W', 'T'}, {'C', 'F'}])
        '(CF)(WT)'
    """
    # Special case: all singletons
    if all(len(coalition) == 1 for coalition in partition):
        return '( )'

    # Filter out singletons and format non-singleton coalitions
    non_singletons = [coalition for coalition in partition if len(coalition) > 1]

    if not non_singletons:
        return '( )'

    # Sort non-singleton coalitions: by size (descending), then alphabetically by first member
    sorted_non_singletons = sorted(non_singletons,
                                   key=lambda c: (-len(c), ''.join(sorted(c))))

    # Format each coalition
    formatted = ['(' + ''.join(sorted(coalition)) + ')' for coalition in sorted_non_singletons]

    return ''.join(formatted)


def generate_coalition_structures(players: List[str]) -> List[str]:
    """
    Generate all coalition structure names for a list of players.

    Args:
        players: List of player names (e.g., ['W', 'T', 'C', 'F'])

    Returns:
        List of coalition structure names, sorted by convention

    Example:
        >>> generate_coalition_structures(['W', 'T', 'C'])
        ['( )', '(TC)', '(WC)', '(WT)', '(WTC)']
    """
    partitions = generate_set_partitions(players)
    structures = [partition_to_string(p) for p in partitions]

    # Remove duplicates (due to our naming convention) and sort
    unique_structures = sorted(set(structures), key=lambda s: (len(s), s))

    return unique_structures


def partition_to_coalition_map(partition: List[Set[str]], all_players: List[str]):
    """
    Convert a partition to a coalition map for creating State objects.

    Args:
        partition: List of sets representing coalitions
        all_players: List of all player names

    Returns:
        List of lists, where each sublist contains player names in that coalition

    Example:
        >>> partition_to_coalition_map([{'W', 'T'}, {'C'}], ['W', 'T', 'C'])
        [['W', 'T'], ['C']]
    """
    # Sort coalitions by size (descending), then alphabetically
    sorted_coalitions = sorted(partition, key=lambda c: (-len(c), sorted(c)))

    return [sorted(list(coalition)) for coalition in sorted_coalitions]


def generate_all_coalition_maps(players: List[str]) -> dict:
    """
    Generate coalition maps for all possible coalition structures.

    Args:
        players: List of player names

    Returns:
        Dictionary mapping structure names to lists of coalitions

    Example:
        >>> generate_all_coalition_maps(['W', 'T', 'C'])
        {
            '( )': [['W'], ['T'], ['C']],
            '(TC)': [['T', 'C'], ['W']],
            ...
        }
    """
    partitions = generate_set_partitions(players)
    result = {}

    for partition in partitions:
        name = partition_to_string(partition)
        if name not in result:  # Avoid duplicates due to naming convention
            result[name] = partition_to_coalition_map(partition, players)

    return result


if __name__ == '__main__':
    # Test with n=3
    print("n=3 players (W, T, C):")
    players_3 = ['W', 'T', 'C']
    structures_3 = generate_coalition_structures(players_3)
    print(f"Number of structures: {len(structures_3)}")
    print(f"Structures: {structures_3}")

    maps_3 = generate_all_coalition_maps(players_3)
    print("\nCoalition maps:")
    for name, coalitions in sorted(maps_3.items()):
        print(f"  {name}: {coalitions}")

    # Test with n=4
    print("\n" + "="*60)
    print("n=4 players (W, T, C, F):")
    players_4 = ['W', 'T', 'C', 'F']
    structures_4 = generate_coalition_structures(players_4)
    print(f"Number of structures: {len(structures_4)}")
    print(f"Structures: {structures_4}")

    maps_4 = generate_all_coalition_maps(players_4)
    print("\nCoalition maps:")
    for name, coalitions in sorted(maps_4.items()):
        print(f"  {name}: {coalitions}")
