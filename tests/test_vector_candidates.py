import unittest

from app.storage.vector_candidates import ranked_candidates


class VectorCandidateContainerTests(unittest.TestCase):
    def test_malformed_outer_containers_are_discarded(self):
        for malformed in (1, True, "unexpected", {"0": []}):
            for field in ("ids", "distances"):
                with self.subTest(field=field, malformed=malformed):
                    results = {"ids": [["belief-1"]], "distances": [[0.2]]}
                    results[field] = malformed
                    self.assertEqual(ranked_candidates(results, 5), [])

    def test_valid_candidates_keep_scores(self):
        self.assertEqual(
            ranked_candidates({"ids": [["belief-1"]], "distances": [[0.2]]}, 5),
            [("belief-1", 0.2)],
        )
