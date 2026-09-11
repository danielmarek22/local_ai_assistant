import math
from itertools import islice


def ranked_candidates(results: dict, limit: int) -> list[tuple[str, float]]:
    """Read bounded candidate IDs/scores; vector text and metadata are not authority."""
    ids = results.get("ids") or []
    distances = results.get("distances") or []
    if not ids or not distances:
        return []
    if not isinstance(ids[0], list) or not isinstance(distances[0], list):
        return []
    candidates = []
    seen = set()
    for record_id, distance in islice(zip(ids[0], distances[0]), limit):
        if (not isinstance(record_id, str) or not record_id or record_id in seen
                or isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not math.isfinite(distance)):
            continue
        seen.add(record_id)
        candidates.append((record_id, float(distance)))
    return candidates
