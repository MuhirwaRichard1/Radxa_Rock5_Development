#!/usr/bin/env python3
"""
Interactive chat with DeepSeek-R1-Distill-Qwen-1.5B on RK3588 NPU via RKLLM.

Uses rkllm_createDefaultParam for proper model defaults and the library's
internal chat template with enable_thinking support.

IMPORTANT: This model requires RKLLM runtime v1.2.2. Using v1.2.3 causes
severe PAD token degeneration (repetitive output).

Usage:
    cd ~/vision && source vision_venv/bin/activate && python3 -u deepseek_chat.py
"""

import ctypes
import re
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rkllm_wrapper import (
    RKLLMParam, RKLLMInput, RKLLMInferParam, RKLLMResult,
    RKLLM_Handle_t, RKLLM_INPUT_PROMPT, RKLLM_INFER_GENERATE,
    LLMCallState_NORMAL, LLMCallState_FINISH, LLMCallState_ERROR,
    _CallbackType,
)

MODEL_PATH = (
    "/home/radxa/.cache/modelscope/hub/models/radxa/"
    "DeepSeek-R1-Distill-Qwen-1___5B_RKLLM/"
    "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"
)
# IMPORTANT: Must use v1.2.2 runtime — v1.2.3 causes PAD token degeneration
LIB_PATH = (
    "/home/radxa/.cache/modelscope/hub/models/radxa/"
    "DeepSeek-R1-Distill-Qwen-1___5B_RKLLM/"
    "demo_Linux_aarch64/lib/librkllmrt.so"
)


class DeepSeekChat:
    """RKLLM chat with streaming output for DeepSeek-R1."""

    def __init__(self, model_path, lib_path):
        self._token_buf = ""
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._streaming = False

        self._callback = _CallbackType(self._on_token)
        self._lib = ctypes.CDLL(lib_path)

        # Use library defaults for proper parameter initialization
        self._lib.rkllm_createDefaultParam.restype = RKLLMParam
        param = self._lib.rkllm_createDefaultParam()

        param.model_path = model_path.encode()
        param.max_context_len = 4096
        param.max_new_tokens = 1024
        param.top_k = 1
        param.skip_special_token = True
        param.extend_param.base_domain_id = 1

        self._handle = RKLLM_Handle_t()
        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), _CallbackType,
        ]
        self._lib.rkllm_init.restype = ctypes.c_int
        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param), self._callback
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed: {ret}")

        # Let the library use its internal chat template + enable_thinking

        self._lib.rkllm_run.argtypes = [
            RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int

        self._lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_abort.restype = ctypes.c_int

        self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_destroy.restype = ctypes.c_int

        self._infer = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self._infer), 0, ctypes.sizeof(RKLLMInferParam))
        self._infer.mode = RKLLM_INFER_GENERATE
        self._infer.keep_history = 0

        print("Model loaded.\n")

    def _on_token(self, result, userdata, state):
        if state == LLMCallState_NORMAL:
            tok = result.contents.text.decode("utf-8", errors="replace")
            with self._lock:
                self._token_buf += tok
            if self._streaming:
                sys.stdout.write(tok)
                sys.stdout.flush()
        elif state in (LLMCallState_FINISH, LLMCallState_ERROR):
            if state == LLMCallState_ERROR:
                sys.stderr.write("\n[inference error]\n")
            self._finished.set()

    def chat(self, user_msg, stream=True):
        """Run single-turn chat inference with streaming."""
        with self._lock:
            self._token_buf = ""
        self._finished.clear()
        self._streaming = stream

        inp = RKLLMInput()
        ctypes.memset(ctypes.byref(inp), 0, ctypes.sizeof(RKLLMInput))
        inp.input_type = RKLLM_INPUT_PROMPT
        inp.role = b"user"
        inp.input_data.prompt_input = user_msg.encode()

        self._lib.rkllm_run(
            self._handle, ctypes.byref(inp), ctypes.byref(self._infer), None,
        )
        self._finished.wait()

        with self._lock:
            return self._token_buf

    def close(self):
        try:
            self._lib.rkllm_destroy(self._handle)
        except Exception:
            pass


def clean_response(text):
    """Strip <think> blocks for clean display."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def main():
    print("Loading DeepSeek-R1-Distill-Qwen-1.5B on NPU...")
    llm = DeepSeekChat(MODEL_PATH, LIB_PATH)

    print("Chat with DeepSeek-R1 (1.5B). Type 'quit' to exit.\n")

    try:
        while True:
            try:
                user_input = input("You> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            print("DeepSeek> ", end="", flush=True)
            raw = llm.chat(user_input, stream=True)
            print()

            cleaned = clean_response(raw)
            if cleaned and cleaned != raw.strip():
                print(f"\n  Answer: {cleaned}")

            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        llm.close()
        print("Model unloaded.")


if __name__ == "__main__":
    main()
