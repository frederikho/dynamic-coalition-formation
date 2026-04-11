"""Shared constants for ordinal-ranking search."""

# When n_states! exceeds this, don't try to enumerate all permutations.
# Use random sampling instead.  9! = 362880 fits in ~3 MB; anything larger
# risks OOM and is never exhaustively searchable anyway.
LARGE_PERM_THRESHOLD = 362880  # 9!
