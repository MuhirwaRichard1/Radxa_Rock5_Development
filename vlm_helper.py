#!/usr/bin/env python3
"""
VLM helper: vision encoder (RKNN) + language model (RKLLM) for Qwen2-VL-2B.

Usage:
    vlm = VLMHelper()
    answer = vlm.describe(frame)
    answer = vlm.describe(frame, "What color is this object?")
    vlm.close()
"""

import numpy as np
from rknnlite.api import RKNNLite

from rkllm_wrapper import RKLLMWrapper

# Model paths
VISION_RKNN = "/home/radxa/Desktop/multimodal_model_demo/rknn/qwen2_vl_2b_vision_rk3588.rknn"
RKLLM_MODEL = "/home/radxa/Desktop/multimodal_model_demo/rkllm/qwen2-vl-2b-instruct_w8a8_rk3588.rkllm"
RKLLM_LIB = "/home/radxa/Desktop/multimodal_model_demo/deploy/install/demo_Linux_aarch64/lib/librkllmrt.so"

# Vision encoder output: 196 tokens x 1536 hidden dim
N_IMAGE_TOKENS = 196
IMAGE_SIZE = 392


def _build_prompt(question):
    """Build multimodal prompt with <image> placeholder.

    The RKLLM library handles the chat template internally when
    img_start/img_end/img_content are set in RKLLMParam.
    We just use <image> as the placeholder (same as the C++ demo).
    """
    return f"<image>{question}"


class VLMHelper:
    """Loads vision encoder + RKLLM once, provides describe(frame, question) method."""

    def __init__(self):
        print("Loading vision encoder RKNN...")
        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(VISION_RKNN)
        if ret != 0:
            raise RuntimeError(f"Failed to load vision RKNN model: {ret}")

        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            self._rknn.release()
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        print("Vision encoder ready.")

        print("Loading RKLLM language model...")
        self._llm = RKLLMWrapper(RKLLM_MODEL, RKLLM_LIB, max_new_tokens=256)
        print("VLM ready.")

    def _preprocess(self, frame):
        """Preprocess BGR OpenCV frame for vision encoder.

        The C++ demo passes NHWC uint8 RGB to the RKNN encoder.  The RKNN
        model was compiled with mean/std normalization baked in, so the
        runtime handles normalization and layout conversion internally.
        """
        import cv2

        # Expand to square with gray fill (matches C++ demo)
        h, w = frame.shape[:2]
        if h != w:
            size = max(h, w)
            square = np.full((size, size, 3), 128, dtype=np.uint8)
            y_off = (size - h) // 2
            x_off = (size - w) // 2
            square[y_off:y_off+h, x_off:x_off+w] = frame
            frame = square

        # Resize to 392x392, BGR→RGB, keep as uint8 NHWC
        img = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0)  # (1, 392, 392, 3) uint8

    def _run_vision(self, frame):
        """Run vision encoder on a frame, return (n_tokens, hidden_dim) features."""
        img = self._preprocess(frame)
        outputs = self._rknn.inference(inputs=[img])
        if outputs is None:
            raise RuntimeError("Vision encoder inference failed")
        # Expected shape: (196, 1536) or (1, 196, 1536)
        features = outputs[0]
        if features.ndim == 3:
            features = features[0]
        return features.astype(np.float32)

    def describe(self, frame, question="Describe what you see in this image briefly."):
        """Send a frame + question to the VLM and return the answer string."""
        features = self._run_vision(frame)
        prompt = _build_prompt(question)
        return self._llm.run_multimodal(prompt, features, N_IMAGE_TOKENS)

    def close(self):
        """Release all resources."""
        try:
            self._rknn.release()
        except Exception:
            pass
        try:
            self._llm.close()
        except Exception:
            pass
