import unittest

from app.core.turn_input import TurnInput
from app.perception.attachments import Attachment, ImageAttachment


class TurnInputTests(unittest.TestCase):
    def test_image_attachments_keep_image_wording(self):
        turn_input = TurnInput(
            user_text="",
            attachments=[
                ImageAttachment(
                    name="clipboard.png",
                    mime_type="image/png",
                    base64_data="aGVsbG8=",
                    size_bytes=5,
                )
            ],
        )

        self.assertEqual(
            turn_input.retrieval_text(),
            "user shared image attachments: clipboard.png",
        )
        self.assertEqual(turn_input.history_text(), "[User attached 1 image]")

    def test_non_image_attachments_use_file_wording(self):
        turn_input = TurnInput(
            user_text="",
            attachments=[
                Attachment(
                    name="notes.pdf",
                    mime_type="application/pdf",
                    size_bytes=42,
                )
            ],
        )

        self.assertEqual(
            turn_input.retrieval_text(),
            "user shared file attachments: notes.pdf",
        )
        self.assertEqual(turn_input.history_text(), "[User attached 1 file]")


if __name__ == "__main__":
    unittest.main()
