import unittest

from app.beliefs.subjects import (
    default_allowed_subjects, grounded_subject_reference, participant_subject,
    resolve_subject_reference,
)


class BeliefSubjectGroundingTests(unittest.TestCase):
    def setUp(self):
        self.subjects = default_allowed_subjects("sender", "Alice", "human")
        self.sender = self.subjects[0]

    def test_english_and_polish_explicit_self_references_agree_with_validator(self):
        for reference in ("I", "I'm", "i’m", "I'll", "My", "me", "myself", "Ja", "mnie",
                          "mi", "mną", "Mój", "moja", "Moje", "moi", "mojego", "mojej",
                          "mojemu", "moją", "moim", "moich", "moimi"):
            with self.subTest(reference=reference):
                text = f"{reference} ..."
                offered = grounded_subject_reference(text, self.sender, self.subjects, "sender")
                self.assertIsNotNone(offered)
                self.assertEqual(resolve_subject_reference(offered, text, self.subjects, "sender"), self.sender)
                self.assertEqual(resolve_subject_reference(reference, text, self.subjects, "sender"), self.sender)

    def test_substrings_and_lowercase_i_cannot_impersonate_sender(self):
        for text, reference in (("Luna i Oreo", "i"), ("IKEA", "I"),
                                ("myth", "my"), ("jamnik", "ja"), ("Aliexpress", "Ali")):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    resolve_subject_reference(reference, text, self.subjects, "sender")
                self.assertIsNone(grounded_subject_reference(text, self.sender, self.subjects, "sender"))

    def test_catalog_omits_partial_ambiguous_and_second_person_names(self):
        cases = (
            ("Annabelle is here", "Ann", []),
            ("Sam is here", "Sam", [participant_subject("other", "Sam", "human")]),
            ("You are here", "You", []),
        )
        for text, name, others in cases:
            with self.subTest(name=name):
                subject = participant_subject("sender", name, "human")
                self.assertIsNone(grounded_subject_reference(text, subject, [subject, *others], "sender"))

    def test_named_other_participant_keeps_its_authoritative_identity(self):
        bob = participant_subject("bob", "Bob", "human")
        subjects = [*self.subjects, bob]
        text = "Bob is testing Astra."
        offered = grounded_subject_reference(text, bob, subjects, "sender")
        self.assertEqual(offered, "Bob")
        self.assertEqual(resolve_subject_reference(offered, text, subjects, "sender"), bob)
