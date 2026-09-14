import unittest

import requests

from app.llm import image_fallback


def _http_error(status: int, body: bytes) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body
    return requests.HTTPError(response=response)


class ImageFallbackPolicyTests(unittest.TestCase):
    def test_image_requests_disable_transport_retries_unless_overridden(self):
        messages = [{"role": "user", "content": "look", "images": ["one"]}]

        self.assertEqual(image_fallback.resolve_request_retries(messages, None), 0)
        self.assertEqual(image_fallback.resolve_request_retries(messages, 3), 3)
        self.assertIsNone(
            image_fallback.resolve_request_retries(
                [{"role": "user", "content": "hello"}],
                None,
            )
        )

    def test_retry_requires_images_supported_status_and_image_error(self):
        messages = [{"role": "user", "content": "look", "images": ["one"]}]
        invalid_image = _http_error(400, b'{"error":"invalid image data"}')

        self.assertTrue(
            image_fallback.should_retry_without_images(
                invalid_image,
                messages,
                multimodal_supported=True,
            )
        )
        self.assertFalse(
            image_fallback.should_retry_without_images(
                invalid_image,
                messages,
                multimodal_supported=False,
            )
        )
        self.assertFalse(
            image_fallback.should_retry_without_images(
                _http_error(500, b'{"error":"invalid image data"}'),
                messages,
                multimodal_supported=True,
            )
        )
        self.assertFalse(
            image_fallback.should_retry_without_images(
                _http_error(400, b'{"error":"invalid request"}'),
                messages,
                multimodal_supported=True,
            )
        )

    def test_fallback_candidates_prioritize_current_images_without_mutation(self):
        messages = [
            {"role": "user", "content": "old", "images": ["old-image"]},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "current", "images": ["one", "two"]},
        ]

        candidates = image_fallback.build_fallback_messages(messages)

        self.assertEqual(
            [(strategy, dropped) for _, strategy, dropped in candidates],
            [
                ("without image 1 from message 3", 1),
                ("without image 2 from message 3", 1),
                ("without current message images", 2),
                ("without images from message 1", 0),
                ("without all images", 2),
            ],
        )
        self.assertEqual(candidates[0][0][2]["images"], ["two"])
        self.assertNotIn("images", candidates[-1][0][0])
        self.assertNotIn("images", candidates[-1][0][2])
        self.assertEqual(messages[0]["images"], ["old-image"])
        self.assertEqual(messages[2]["images"], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
