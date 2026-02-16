#!/usr/bin/env python3
"""Minimal test to debug RKLLM struct layout issues."""

import ctypes
import sys
import threading
import numpy as np

LIB_PATH = "/home/radxa/Desktop/multimodal_model_demo/deploy/install/demo_Linux_aarch64/lib/librkllmrt.so"
MODEL_PATH = "/home/radxa/Desktop/multimodal_model_demo/rkllm/qwen2-vl-2b-instruct_w8a8_rk3588.rkllm"
VISION_RKNN = "/home/radxa/Desktop/multimodal_model_demo/rknn/qwen2_vl_2b_vision_rk3588.rknn"

# Load library
lib = ctypes.CDLL(LIB_PATH)

# --- Minimal struct definitions ---
RKLLM_Handle_t = ctypes.c_void_p

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

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
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

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]

# --- Try BOTH struct layouts ---

class RKLLMInput(ctypes.Structure):
    """Correct layout from disassembly: role first, then reserved, then input_type, then union."""
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("_reserved", ctypes.c_int32),
        ("input_type", ctypes.c_int32),
        ("input_data", RKLLMInputUnion),
    ]

print(f"RKLLMInput size: {ctypes.sizeof(RKLLMInput)}")
print(f"RKLLMMultiModelInput size: {ctypes.sizeof(RKLLMMultiModelInput)}")
print(f"RKLLMInputUnion size: {ctypes.sizeof(RKLLMInputUnion)}")
print(f"RKLLMParam size: {ctypes.sizeof(RKLLMParam)}")
print(f"RKLLMInferParam size: {ctypes.sizeof(RKLLMInferParam)}")

# Test rkllm_createDefaultParam to validate RKLLMParam layout
lib.rkllm_createDefaultParam.restype = RKLLMParam
lib.rkllm_createDefaultParam.argtypes = []
default_param = lib.rkllm_createDefaultParam()
print(f"\nDefault param from library:")
print(f"  max_context_len: {default_param.max_context_len}")
print(f"  max_new_tokens:  {default_param.max_new_tokens}")
print(f"  top_k:           {default_param.top_k}")
print(f"  top_p:           {default_param.top_p}")
print(f"  temperature:     {default_param.temperature}")
print(f"  repeat_penalty:  {default_param.repeat_penalty}")
print(f"  skip_special_token: {default_param.skip_special_token}")

# Setup callback
output_text = ""
finished = threading.Event()

CallbackType = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

def callback_impl(result, userdata, state):
    global output_text
    if state == 0:  # NORMAL
        token = result.contents.text.decode("utf-8", errors="replace")
        output_text += token
        print(token, end="", flush=True)
    elif state == 2:  # FINISH
        print()
        finished.set()
    elif state == 3:  # ERROR
        print("\n[ERROR]")
        finished.set()

callback = CallbackType(callback_impl)

# Init model using createDefaultParam
print("\n--- Initializing RKLLM model ---")
param = lib.rkllm_createDefaultParam()
param.model_path = MODEL_PATH.encode("utf-8")
param.max_context_len = 4096
param.max_new_tokens = 128
param.top_k = 1
param.skip_special_token = True
param.img_start = "<|vision_start|>".encode("utf-8")
param.img_end = "<|vision_end|>".encode("utf-8")
param.img_content = "<|image_pad|>".encode("utf-8")
param.extend_param.base_domain_id = 1
param.extend_param.enabled_cpus_num = 4
param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

handle = RKLLM_Handle_t()
lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), CallbackType]
lib.rkllm_init.restype = ctypes.c_int
ret = lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), callback)
print(f"rkllm_init returned: {ret}")
if ret != 0:
    sys.exit(1)

# Setup run function
lib.rkllm_run.restype = ctypes.c_int

infer_params = RKLLMInferParam()
ctypes.memset(ctypes.byref(infer_params), 0, ctypes.sizeof(RKLLMInferParam))
infer_params.mode = 0  # GENERATE
infer_params.keep_history = 0

# --- Test text prompt with correct layout ---
print("\n--- Test: Text prompt ---")
output_text = ""
finished.clear()

lib.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]

inp = RKLLMInput()
ctypes.memset(ctypes.byref(inp), 0, ctypes.sizeof(RKLLMInput))
inp.role = b"user"
inp.input_type = 0  # RKLLM_INPUT_PROMPT
inp.input_data.prompt_input = b"What is 2+2? Answer briefly."

print("Calling rkllm_run...", flush=True)
ret = lib.rkllm_run(handle, ctypes.byref(inp), ctypes.byref(infer_params), None)
print(f"rkllm_run returned: {ret}")
if ret == 0:
    finished.wait(timeout=30)
    print(f"SUCCESS! Output: '{output_text}'")

print("\n--- Cleaning up ---")
lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
lib.rkllm_destroy.restype = ctypes.c_int
lib.rkllm_destroy(handle)
print("Done!")
