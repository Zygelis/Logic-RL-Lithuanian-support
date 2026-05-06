#!/usr/bin/env python3
"""Minimal single-process test to load the model and run a short generation using Transformers/vLLM attention backend settings.
Run this outside Ray to isolate vLLM/FlashAttention/NCCL issues.
"""
import os
import traceback
import torch

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print("Could not import transformers:", e)
        raise

    print("Torch CUDA available:", torch.cuda.is_available(), "cuda_version:", torch.version.cuda)

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print("Loading model with device_map=auto, torch_dtype=bfloat16, low_cpu_mem_usage=True")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        p = next(model.parameters())
        print("First param dtype:", p.dtype, "device:", p.device)

        # Ensure model is on CUDA if available
        if torch.cuda.is_available():
            try:
                model.to("cuda")
                print("Moved model to cuda, first param device:", next(model.parameters()).device)
            except Exception as e:
                print("Failed to move model to cuda:", e)

        print("Running a tiny generation test...")
        text = "Sveiki. Koks yra rezultatas?"
        inputs = tokenizer(text, return_tensors="pt")
        # move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            # Use autocast when CUDA available and using flash-attn
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.generate(**inputs, max_new_tokens=16)
            else:
                out = model.generate(**inputs, max_new_tokens=16)

            print("Generation OK:\n", tokenizer.decode(out[0], skip_special_tokens=True))
        except Exception as e:
            print("Generation failed:")
            traceback.print_exc()

    except Exception:
        print("Model load/generation failed:\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()
