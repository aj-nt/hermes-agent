"""Tests for agent.kore.vision_utils -- extracted vision utility functions."""

import os
import tempfile
from pathlib import Path

import pytest

from agent.kore.vision_utils import (
    content_has_image_parts,
    materialize_data_url_for_vision,
)


class TestContentHasImageParts:

    def test_list_with_image_url(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        assert content_has_image_parts(content) is True

    def test_list_with_input_image(self):
        content = [{"type": "input_image", "url": "data:image/png;base64,abc"}]
        assert content_has_image_parts(content) is True

    def test_list_without_images(self):
        content = [{"type": "text", "text": "hello"}]
        assert content_has_image_parts(content) is False

    def test_empty_list(self):
        assert content_has_image_parts([]) is False

    def test_non_list(self):
        assert content_has_image_parts("hello") is False
        assert content_has_image_parts(None) is False


class TestMaterializeDataUrlForVision:

    def test_basic_data_url(self):
        import base64
        data = base64.b64encode(b"fake image data").decode()
        url = f"data:image/png;base64,{data}"
        path_str, path_obj = materialize_data_url_for_vision(url)
        assert isinstance(path_str, str)
        assert isinstance(path_obj, Path)
        assert path_obj.exists()
        # Clean up
        path_obj.unlink(missing_ok=True)

    def test_jpeg_suffix(self):
        import base64
        data = base64.b64encode(b"fake jpeg data").decode()
        url = f"data:image/jpeg;base64,{data}"
        _, path_obj = materialize_data_url_for_vision(url)
        assert path_obj.suffix == ".jpg"
        path_obj.unlink(missing_ok=True)


class TestVisionUtilsBackwardCompat:

    @pytest.fixture(autouse=True)
    def _make_agent(self):
        from run_agent import AIAgent
        self.agent = AIAgent.__new__(AIAgent)

    def test_content_has_image_parts_compat(self):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        assert content_has_image_parts(content) == self.agent._content_has_image_parts(content)

    def test_content_has_no_images_compat(self):
        content = [{"type": "text", "text": "hello"}]
        assert content_has_image_parts(content) == self.agent._content_has_image_parts(content)
