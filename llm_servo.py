#!/usr/bin/env python3
"""
LLM Servo Controller — natural language pan-tilt control via DeepSeek-R1 on NPU.

Type commands like "look left", "tilt up a bit", "center" and the LLM
interprets them as pan/tilt servo angles.

Uses DeepSeek-R1-Distill-Qwen-1.5B with RKLLM runtime v1.2.2.

Requires: periphery (PWM servos), rkllm_wrapper (ctypes structs).
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
from servo_pan_tilt_tracker_fast import Servo

# --- Paths ---
MODEL_PATH = (
    "/home/radxa/.cache/modelscope/hub/models/radxa/"
    "DeepSeek-R1-Distill-Qwen-1___5B_RKLLM/"
    "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"
)
# Must use v1.2.2 runtime — v1.2.3 causes PAD token degeneration
LIB_PATH = (
    "/home/radxa/.cache/modelscope/hub/models/radxa/"
    "DeepSeek-R1-Distill-Qwen-1___5B_RKLLM/"
    "demo_Linux_aarch64/lib/librkllmrt.so"
)

# --- Servo config ---
PAN_CHIP = 4
PAN_CHANNEL = 0
TILT_CHIP = 3
TILT_CHANNEL = 0
SERVO_FREQ = 50
TILT_MIN = 30.0
TILT_MAX = 150.0

# --- Defaults ---
CENTER = 90.0
PAN_MIN = 0.0
PAN_MAX = 180.0

SYSTEM_PROMPT = (
    "Convert camera commands to pan,tilt. "
    "Pan 0-180: 0=right, 90=center, 180=left. "
    "Tilt 30-150: 30=down, 90=center, 150=up. "
    "Answer with ONLY two numbers like: 135,90"
)

# Match two numbers separated by comma (handles LaTeX \boxed{X, Y} and plain X,Y)
NUM_PAIR_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*\\?\s*(-?\d+(?:\.\d+)?)")


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class ServoLLM:
    """DeepSeek-R1 on RKLLM v1.2.2 with custom chat template for servo control."""

    def __init__(self, model_path, lib_path):
        self._token_buf = ""
        self._finished = threading.Event()
        self._lock = threading.Lock()

        self._callback = _CallbackType(self._on_token)
        self._lib = ctypes.CDLL(lib_path)

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
        # The model needs thinking mode to reason through servo commands

        self._lib.rkllm_run.argtypes = [
            RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int

        self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_destroy.restype = ctypes.c_int

        self._infer = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self._infer), 0, ctypes.sizeof(RKLLMInferParam))
        self._infer.mode = RKLLM_INFER_GENERATE
        self._infer.keep_history = 0

    def _on_token(self, result, userdata, state):
        if state == LLMCallState_NORMAL:
            tok = result.contents.text.decode("utf-8", errors="replace")
            with self._lock:
                self._token_buf += tok
        elif state in (LLMCallState_FINISH, LLMCallState_ERROR):
            self._finished.set()

    def infer(self, user_msg):
        """Send user message, return raw response text."""
        with self._lock:
            self._token_buf = ""
        self._finished.clear()

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


def parse_response(text):
    """Extract (pan, tilt) from model output.

    Looks for \\boxed{X, Y} first, then falls back to last X,Y in the
    post-think section, then anywhere in the text.
    """
    # Try \boxed{X, Y} first (DeepSeek-R1 style)
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        m = NUM_PAIR_RE.search(boxed.group(1))
        if m:
            p = clamp(float(m.group(1)), PAN_MIN, PAN_MAX)
            t = clamp(float(m.group(2)), TILT_MIN, TILT_MAX)
            return p, t

    # Try post-think section
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    matches = list(NUM_PAIR_RE.finditer(clean))
    if matches:
        m = matches[-1]  # Last match (most likely the final answer)
        p = clamp(float(m.group(1)), PAN_MIN, PAN_MAX)
        t = clamp(float(m.group(2)), TILT_MIN, TILT_MAX)
        return p, t

    # Fallback: search entire text, take last match
    matches = list(NUM_PAIR_RE.finditer(text))
    if matches:
        m = matches[-1]
        p = clamp(float(m.group(1)), PAN_MIN, PAN_MAX)
        t = clamp(float(m.group(2)), TILT_MIN, TILT_MAX)
        return p, t

    return None


def main():
    print("Loading DeepSeek-R1 model on NPU (v1.2.2 runtime)...")
    llm = ServoLLM(MODEL_PATH, LIB_PATH)
    print("Model loaded.")

    print("Initializing servos...")
    pan_servo = Servo(PAN_CHIP, PAN_CHANNEL, SERVO_FREQ)
    tilt_servo = Servo(TILT_CHIP, TILT_CHANNEL, SERVO_FREQ,
                       min_angle=TILT_MIN, max_angle=TILT_MAX)

    pan_angle = CENTER
    tilt_angle = CENTER
    pan_servo.move_to(pan_angle)
    tilt_servo.move_to(tilt_angle)

    print(f"\nServos centered at pan={pan_angle:.0f}, tilt={tilt_angle:.0f}")
    print("Commands: natural language, 'reset', 'quit'")
    print("Examples: 'look left', 'tilt up a bit', 'pan right far', 'center'\n")

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
            if user_input.lower() == "reset":
                pan_angle = CENTER
                tilt_angle = CENTER
                pan_servo.move_to(pan_angle)
                tilt_servo.move_to(tilt_angle)
                print(f"  [reset] pan={pan_angle:.0f}, tilt={tilt_angle:.0f}\n")
                continue

            msg = (
                f"You control a camera with pan and tilt servos. "
                f"Pan: 0=right, 90=center, 180=left. "
                f"Tilt: 30=down, 90=center, 150=up. "
                f"Current: pan={int(pan_angle)}, tilt={int(tilt_angle)}. "
                f"Command: \"{user_input}\". "
                f"What are the new pan,tilt values? Answer with just two numbers like 135,90"
            )
            print("  thinking...", end="", flush=True)
            response = llm.infer(msg)
            print("\r           \r", end="")

            result = parse_response(response)
            if result:
                new_pan, new_tilt = result
                pan_servo.move_to(new_pan)
                tilt_servo.move_to(new_tilt)
                pan_angle = new_pan
                tilt_angle = new_tilt
                # Show cleaned response
                clean = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                print(f"  LLM: {clean}")
                print(f"  -> pan={pan_angle:.0f}, tilt={tilt_angle:.0f}\n")
            else:
                # Show first 200 chars of response for debugging
                raw = response.strip()[:200]
                print(f"  LLM: {raw}")
                print("  (could not parse angles, servos unchanged)\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pan_servo.close()
        tilt_servo.close()
        llm.close()
        print("Servos released, model unloaded.")


if __name__ == "__main__":
    main()
