"""Small test fixtures shared by related suites."""

import json
from unittest.mock import Mock

import requests


def consume_generator(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as completed:
            return events, completed.value


def make_response(*, data=None, lines=None, status_code=200, http_error=None):
    """Build a readable, closable response; tests may replace iter_lines to fail midstream."""
    response = requests.Response()
    response.status_code = status_code
    response._content = (
        b"\n".join(lines) if lines is not None
        else json.dumps(data if data is not None else {}).encode("utf-8")
    )
    response._content_consumed = True
    if http_error is not None:
        response.raise_for_status = Mock(side_effect=http_error)
    return response
