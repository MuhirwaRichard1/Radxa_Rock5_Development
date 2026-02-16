#!/usr/bin/env python3
"""
Reusable RKLLM ctypes wrapper for Rockchip NPU LLM inference.

Supports text-only and multimodal (vision+text) inference modes.
Extracted from the rkllm_server flask_server.py example.
"""

import ctypes
import sys
import threading

# --- ctypes structure definitions ---

RKLLM_Handle_t = ctypes.c_void_p

LLMCallState_NORMAL = 0
LLMCallState_WAITING = 1
LLMCallState_FINISH = 2
LLMCallState_ERROR = 3

RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 106),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]


class RKLLMInput(ctypes.Structure):
    # Layout reverse-engineered from the compiled C++ demo binary (v1.2.3).
    # role comes FIRST, then a reserved int32, then input_type, then the union.
    _fields_ = [
        ("role", ctypes.c_char_p),          # offset 0  (8 bytes)
        ("_reserved", ctypes.c_int32),      # offset 8  (4 bytes, always 0)
        ("input_type", ctypes.c_int32),     # offset 12 (4 bytes)
        ("input_data", RKLLMInputUnion),    # offset 16 (48 bytes)
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
    ]


# Callback function type: (result_ptr, userdata, state) -> None
_CallbackType = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)


class RKLLMWrapper:
    """Clean wrapper around librkllmrt.so for text and multimodal inference."""

    def __init__(self, model_path, lib_path, max_new_tokens=256):
        # Accumulated output and synchronization
        self._output_text = ""
        self._finished = threading.Event()
        self._lock = threading.Lock()

        # Keep reference to prevent GC
        self._callback = _CallbackType(self._callback_impl)

        # Load library
        self._lib = ctypes.CDLL(lib_path)

        # Set up model parameters
        param = RKLLMParam()
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = 4096
        param.max_new_tokens = max_new_tokens
        param.skip_special_token = True
        param.n_keep = -1
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.is_async = False

        # Qwen2-VL image token markers
        param.img_start = "<|vision_start|>".encode("utf-8")
        param.img_end = "<|vision_end|>".encode("utf-8")
        param.img_content = "<|image_pad|>".encode("utf-8")

        # Use big cores (4-7)
        param.extend_param.base_domain_id = 1
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

        # Init model
        self._handle = RKLLM_Handle_t()

        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            _CallbackType,
        ]
        self._lib.rkllm_init.restype = ctypes.c_int

        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param), self._callback
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")

        # Set up run function
        self._lib.rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int

        # Set up destroy function
        self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_destroy.restype = ctypes.c_int

        # Default infer params
        self._infer_params = RKLLMInferParam()
        ctypes.memset(
            ctypes.byref(self._infer_params), 0, ctypes.sizeof(RKLLMInferParam)
        )
        self._infer_params.mode = RKLLM_INFER_GENERATE
        self._infer_params.keep_history = 0

        print(f"RKLLM model loaded: {model_path}")

    def _callback_impl(self, result, userdata, state):
        if state == LLMCallState_NORMAL:
            token = result.contents.text.decode("utf-8", errors="replace")
            with self._lock:
                self._output_text += token
        elif state == LLMCallState_FINISH:
            self._finished.set()
        elif state == LLMCallState_ERROR:
            print("RKLLM inference error", file=sys.stderr)
            self._finished.set()

    def run_text(self, prompt):
        """Run text-only inference. Returns generated string."""
        with self._lock:
            self._output_text = ""
        self._finished.clear()

        rkllm_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
        rkllm_input.input_type = RKLLM_INPUT_PROMPT
        rkllm_input.role = b"user"
        rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

        self._lib.rkllm_run(
            self._handle,
            ctypes.byref(rkllm_input),
            ctypes.byref(self._infer_params),
            None,
        )
        self._finished.wait()
        with self._lock:
            return self._output_text

    def run_multimodal(self, prompt, image_embed, n_image_tokens):
        """Run multimodal inference with image embeddings.

        Args:
            prompt: Text prompt (with image token placeholders).
            image_embed: numpy float32 array of shape (n_image_tokens, hidden_dim).
            n_image_tokens: Number of image tokens (e.g. 196).

        Returns:
            Generated text string.
        """
        import numpy as np

        with self._lock:
            self._output_text = ""
        self._finished.clear()

        # Flatten to contiguous float32
        embed_flat = np.ascontiguousarray(image_embed.flatten(), dtype=np.float32)
        embed_ptr = embed_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rkllm_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rkllm_input), 0, ctypes.sizeof(RKLLMInput))
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL
        rkllm_input.role = b"user"
        mm = rkllm_input.input_data.multimodal_input
        mm.prompt = prompt.encode("utf-8")
        mm.image_embed = embed_ptr
        mm.n_image_tokens = n_image_tokens
        mm.n_image = 1
        mm.image_width = 392
        mm.image_height = 392

        self._lib.rkllm_run(
            self._handle,
            ctypes.byref(rkllm_input),
            ctypes.byref(self._infer_params),
            None,
        )
        self._finished.wait()
        with self._lock:
            return self._output_text

    def close(self):
        """Release model resources."""
        try:
            self._lib.rkllm_destroy(self._handle)
        except Exception:
            pass
