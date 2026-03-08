"""
Median Rank Aggregation - Simple Baseline for Ranking Aggregation.

For each candidate, collect its position across all input rankings and use
the median position as its aggregate score. Candidates are ranked from
lowest (best) to highest median position.

Why median instead of mean (Borda)?
- Robust to outliers: a single extreme ranking does not dominate the result.
- Simpler intuition: "most of the time, this candidate sits at position k".
- No pairwise matrix needed: O(n*m) vs O(n^2*m) for Kemeny.
"""

import statistics
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union

RankLike = Union[Sequence[str], Mapping[str, int]]


def _as_position_map(ranking: RankLike) -> Dict[str, float]:
    """Convert a ranking to a {item: position} dict (position 1 = best)."""
    if isinstance(ranking, Mapping):
        return {str(k): float(v) for k, v in ranking.items()}
    elif isinstance(ranking, Sequence):
        return {str(x): float(i + 1) for i, x in enumerate(ranking)}
    raise TypeError("ranking must be a sequence or a mapping")


def median_rank_aggregation(
    rankings: Iterable[RankLike],
    *,
    missing: str = "bottom",
) -> List[str]:
    """
    Aggregate multiple rankings using each candidate's median position.

    For each candidate, collect its position (1 = best) from every ranking
    in which it appears.  The aggregate score is the median of those positions;
    ties are broken by mean position, then alphabetically.

    Complexity: O(n * m) where n = number of rankings, m = number of candidates.
    No pairwise matrix is required, making this much faster than Kemeny for
    large inputs.

    Args:
        rankings: Iterable of rankings (lists/tuples of candidate labels, or
                  dicts mapping label -> position).
        missing:  How to handle candidates absent from a ranking:
                  - 'bottom' (default): treat as placed last (position = m + 1)
                  - 'ignore': only use rankings where the candidate appears

    Returns:
        Ranking (best to worst) as a list of candidate labels.

    Example:
        >>> rankings = [
        ...     ['A', 'B', 'C'],
        ...     ['B', 'A', 'C'],
        ...     ['A', 'C', 'B'],
        ... ]
        >>> median_rank_aggregation(rankings)
        ['A', 'B', 'C']
    """
    if missing not in ("bottom", "ignore"):
        raise ValueError("missing must be 'bottom' or 'ignore'")

    ranks = [_as_position_map(r) for r in rankings]
    if not ranks:
        return []

    # Discover all candidates
    all_items: Set[str] = set()
    for r in ranks:
        all_items.update(r.keys())
    if not all_items:
        return []

    positions: Dict[str, List[float]] = {item: [] for item in all_items}

    for r in ranks:
        bottom_pos = (max(r.values()) if r else 0.0) + 1.0
        for item in all_items:
            if item in r:
                positions[item].append(r[item])
            elif missing == "bottom":
                positions[item].append(bottom_pos)
            # else missing == "ignore": skip this ranking for this item

    def _sort_key(item: str) -> Tuple[float, float, str]:
        pos_list = positions[item]
        if not pos_list:
            # Never appeared and missing='ignore': send to very end
            return (float("inf"), float("inf"), item)
        med = statistics.median(pos_list)
        mean = sum(pos_list) / len(pos_list)  # tiebreaker
        return (med, mean, item)              # alphabetical final tiebreak

    return sorted(all_items, key=_sort_key)
