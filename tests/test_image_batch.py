"""Tests for the single ``images`` batch input on ``generate()``.

The fork replaced the four ``image1..image4`` sockets with one IMAGE batch input.
These drive the real ``generate()`` with a fake ``lmstudio`` module, so the
frames really go through ``convert_image_to_pil`` and the JPEG encode in
``_prepare_images`` - the path that used to fail with
``OSError: cannot write mode RGBA as JPEG`` for 4-channel frames.
"""
import importlib

import numpy as np
import pytest
from PIL import Image

NODE_PACKAGE = "ea_lmstudio_under_test"

_node = importlib.import_module(f"{NODE_PACKAGE}.LMStudio")
EALMStudio = _node.EALMStudio
CUSTOM_MODEL_OPTION = _node.CUSTOM_MODEL_OPTION


class FakeTensor:
    """Just the tensor surface the node touches: shape, slicing, cpu().numpy()."""

    def __init__(self, arr):
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, key):
        return FakeTensor(self.arr[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


def _batch(n, h=8, w=8, c=3, value=0.5):
    return FakeTensor(np.full((n, h, w, c), value, dtype=np.float32))


# --- fake LM Studio --------------------------------------------------------


class _Stats:
    stop_reason = "eosFound"
    tokens_per_second = None
    prompt_tokens_count = None
    predicted_tokens_count = None
    total_tokens_count = None
    time_to_first_token_sec = None
    num_gpu_layers = None
    total_draft_tokens_count = None
    accepted_draft_tokens_count = None
    rejected_draft_tokens_count = None
    used_draft_model_key = None


class _Fragment:
    content = "a caption"
    reasoning_type = "none"
    tokens_count = 1


class _Result:
    content = "a caption"
    stats = _Stats()
    structured = None
    prediction_config = None


class _Stream:
    def __iter__(self):
        yield _Fragment()

    def result(self):
        return _Result()

    def cancel(self):  # pragma: no cover
        pass


class _Model:
    identifier = "vlm"
    model_key = "pub/vlm"

    def respond_stream(self, chat, config=None):
        return _Stream()

    def unload(self):  # pragma: no cover - unload_llm is disabled in these tests
        pass


class _Files:
    """Opens each uploaded file, proving it is a decodable JPEG on disk."""

    def __init__(self):
        self.uploaded = []

    def prepare_image(self, path):
        with Image.open(path) as img:
            img.load()
            self.uploaded.append((img.format, img.mode, img.size))
        return object()


class _LLM:
    def model(self, identifier):
        return _Model()

    def list_loaded(self):
        return [_Model()]


@pytest.fixture
def vlm(monkeypatch):
    """Patch the node's ``lms`` module; return the recorder for uploads/messages."""
    monkeypatch.setattr(
        _node.model_management, "processing_interrupted", lambda: False, raising=False
    )
    monkeypatch.setattr(
        _node.model_management,
        "throw_exception_if_processing_interrupted",
        lambda: None,
        raising=False,
    )

    files = _Files()
    sent = {"image_counts": []}

    class _Client:
        llm = _LLM()

        def __init__(self):
            self.files = files

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _Chat:
        def __init__(self, system_message=None):
            pass

        def add_user_message(self, prompt, images=None):
            sent["image_counts"].append(len(images or []))

    class _FakeLMS:
        Chat = _Chat

        @staticmethod
        def Client(server_address):
            return _Client()

    monkeypatch.setattr(_node, "lms", _FakeLMS)
    return files, sent


def _run(**overrides):
    kwargs = dict(
        system_message="sys",
        prompt="describe",
        model_selection=CUSTOM_MODEL_OPTION,
        custom_model_name="vlm",
        max_tokens=16,
        temperature=0.7,
        seed=0,
        unload_llm=False,
    )
    kwargs.update(overrides)
    return EALMStudio().generate(**kwargs)


def _response(result):
    return result["result"][0]


def _troubleshooting(result):
    return result["result"][2]


# --- input schema ----------------------------------------------------------


def test_node_exposes_one_images_input_not_four_sockets():
    optional = EALMStudio.INPUT_TYPES()["optional"]
    assert optional["images"][0] == "IMAGE"
    for legacy in ("image1", "image2", "image3", "image4"):
        assert legacy not in optional


# --- batch handling --------------------------------------------------------


def test_every_frame_in_the_batch_is_sent_to_the_model(vlm):
    files, sent = vlm

    result = _run(images=_batch(3))

    assert _response(result) == "a caption"
    assert sent["image_counts"] == [3]
    assert len(files.uploaded) == 3
    assert "Image batch received: 3 images" in _troubleshooting(result)
    assert "Total images for VLM: 3" in _troubleshooting(result)


def test_a_single_frame_batch_has_no_batch_banner(vlm):
    files, sent = vlm

    result = _run(images=_batch(1))

    assert sent["image_counts"] == [1]
    assert "Image batch received" not in _troubleshooting(result)


def test_an_unbatched_hwc_tensor_is_treated_as_one_image(vlm):
    files, sent = vlm
    hwc = FakeTensor(np.full((8, 8, 3), 0.5, dtype=np.float32))

    _run(images=hwc)

    assert sent["image_counts"] == [1]
    assert len(files.uploaded) == 1


def test_no_images_input_is_a_plain_text_run(vlm):
    files, sent = vlm

    result = _run()

    assert _response(result) == "a caption"
    assert sent["image_counts"] == [0]
    assert files.uploaded == []


# --- the reported bug ------------------------------------------------------


def test_rgba_frames_are_jpeg_encoded_instead_of_raising(vlm):
    """Regression: 4-channel frames used to fail with
    'cannot write mode RGBA as JPEG', which blanked the response and let a
    non-JSON/None prompt crash the downstream text encoder."""
    files, sent = vlm

    result = _run(images=_batch(2, c=4))

    assert _response(result) == "a caption"
    assert "[ERROR]" not in _troubleshooting(result)
    assert sent["image_counts"] == [2]
    assert [(fmt, mode) for fmt, mode, _ in files.uploaded] == [("JPEG", "RGB")] * 2


def test_mixed_channel_counts_in_one_run_all_upload(vlm):
    """Grayscale, RGB and RGBA frames must each survive the JPEG encode."""
    files, sent = vlm

    for channels in (1, 3, 4):
        _run(images=_batch(1, c=channels))

    assert len(files.uploaded) == 3
    assert all(fmt == "JPEG" for fmt, _, _ in files.uploaded)


def test_resize_applies_to_every_frame(vlm):
    files, sent = vlm

    _run(images=_batch(2, h=2000, w=1000), image_resize="Low (512px)")

    assert all(max(size) == 512 for _, _, size in files.uploaded)
