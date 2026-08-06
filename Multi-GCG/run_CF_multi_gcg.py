import json
import copy
import gc
import time

import torch
import transformers

from CF_multi_gcg import GCGConfig, run_two_models

# =============================================================================
# Configuration
# =============================================================================

QWEN_PATH = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"

JSON_PATH = "./surrogates.json"
OUTPUT_PATH = "./CF_multi_gcg.txt"

WANTED_IDS = {"message_1", "message_4", "message_6", "message_12", "message_13"}

# =============================================================================
# Target (avoid model-specific special tokens to reduce tokenizer mismatch)
# =============================================================================
TARGET = "FUCK YOU! YOU ARE STUPID! I WON'T ANSWER ANY QUESTION!!!"

# Initial optimization string
OPTIM_INIT_STR = "x " * 40


def load_models():
    """Load both models with proper GPU memory allocation."""
    print("=" * 60)
    print("Loading Models (GPU)")
    print("=" * 60)

    print(f"\n[1/2] Loading Mistral: {MISTRAL_PATH}")
    t0 = time.time()

    model_mistral = transformers.AutoModelForCausalLM.from_pretrained(
        MISTRAL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer_mistral = transformers.AutoTokenizer.from_pretrained(
        MISTRAL_PATH,
        use_fast=False,
        trust_remote_code=True,
    )

    # Pad token handling
    if tokenizer_mistral.pad_token is None:
        tokenizer_mistral.pad_token = tokenizer_mistral.eos_token

    print(f"    Loaded in {time.time() - t0:.1f}s")
    print(f"    Device: {model_mistral.device}")

    print(f"\n[2/2] Loading Qwen: {QWEN_PATH}")
    t0 = time.time()

    model_qwen = transformers.AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer_qwen = transformers.AutoTokenizer.from_pretrained(
        QWEN_PATH,
        trust_remote_code=True,
    )
    if tokenizer_qwen.pad_token is None:
        tokenizer_qwen.pad_token = tokenizer_qwen.eos_token

    print(f"    Loaded in {time.time() - t0:.1f}s")
    print(f"    Device: {model_qwen.device}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_qwen, tokenizer_qwen, model_mistral, tokenizer_mistral


def load_data():
    """Load and filter training data."""
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    wanted_ids_str = {str(x) for x in WANTED_IDS}

    multi_messages = [
        d["messages"] for d in data
        if isinstance(d, dict)
        and "id" in d
        and str(d["id"]) in wanted_ids_str
        and isinstance(d.get("messages"), list)
    ]

    selected_ids = [
        str(d["id"]) for d in data
        if isinstance(d, dict) and "id" in d and str(d["id"]) in wanted_ids_str
    ]

    print(f"Selected IDs: {selected_ids}")
    print(f"Total messages: {len(multi_messages)}")

    return multi_messages


def verify_results(
    best_string: str,
    multi_messages: list,
    model_qwen,
    tokenizer_qwen,
    model_mistral,
    tokenizer_mistral,
    num_samples: int = 2
):
    """Verify the optimized string on both models."""
    print("\n" + "=" * 60)
    print("Verification Results")
    print("=" * 60)

    def _ensure_placeholder(msg_list):
        if not any("{optim_str}" in m.get("content", "") for m in msg_list):
            msg_list[-1]["content"] = msg_list[-1].get("content", "") + "{optim_str}"
        return msg_list

    print("\n--- Qwen Outputs ---")
    for i, message_list in enumerate(multi_messages[:num_samples]):
        temp_message_list = copy.deepcopy(message_list)
        temp_message_list = _ensure_placeholder(temp_message_list)
        temp_message_list[-1]["content"] = temp_message_list[-1]["content"].replace(
            "{optim_str}", best_string
        )

        full_prompt = tokenizer_qwen.apply_chat_template(
            temp_message_list, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer_qwen(full_prompt, return_tensors="pt").to(model_qwen.device)

        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            output = model_qwen.generate(**inputs, max_new_tokens=100, do_sample=False)

        print(f"\n[Sample {i+1}]")
        print(tokenizer_qwen.decode(output[0][input_length:]))

    print("\n--- Mistral Outputs ---")
    for i, message_list in enumerate(multi_messages[:num_samples]):
        temp_message_list = copy.deepcopy(message_list)
        temp_message_list = _ensure_placeholder(temp_message_list)
        temp_message_list[-1]["content"] = temp_message_list[-1]["content"].replace(
            "{optim_str}", best_string
        )

        full_prompt = tokenizer_mistral.apply_chat_template(
            temp_message_list, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer_mistral(full_prompt, return_tensors="pt").to(model_mistral.device)

        input_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            output = model_mistral.generate(**inputs, max_new_tokens=100, do_sample=False)

        print(f"\n[Sample {i+1}]")
        print(tokenizer_mistral.decode(output[0][input_length:]))


def main():
    print("\n" + "=" * 60)
    print("Cross-Family GCG Attack (GPU-only)")
    print("=" * 60)

    model_qwen, tokenizer_qwen, model_mistral, tokenizer_mistral = load_models()
    multi_messages = load_data()

    # ==========================================================================
    # GCG Configuration
    # ==========================================================================
    config = GCGConfig(
        num_steps=1500,

        # candidate sampling
        search_width=192,
        batch_size=12,
        topk=128,

        # optimization init
        optim_str_init=OPTIM_INIT_STR,
        filter_ids=False,

        # gradient aggregation
        grad_aggregation="alternate",

        # IMPORTANT: models passed as (model1=mistral, model2=qwen),
        # so primary_model_idx=2 makes Qwen the primary (matches your intent)
        primary_model_idx=2,

        # cache (if your v3 ignores prefix_cache internally, leaving this doesn't hurt)
        use_prefix_cache=True,

        # logging
        log_timing=False,  # GPU timing is async; enable only if you synchronize in v3
        verbosity="INFO",
    )

    print("\n" + "=" * 60)
    print("GCG Configuration")
    print("=" * 60)
    print(f"  num_steps:        {config.num_steps}")
    print(f"  search_width:     {config.search_width}")
    print(f"  batch_size:       {config.batch_size}")
    print(f"  topk:             {config.topk}")
    print(f"  grad_aggregation: {config.grad_aggregation}")
    print(f"  primary_model_idx:{config.primary_model_idx}")
    print(f"  TARGET:           {TARGET[:50]}...")

    print("\n" + "=" * 60)
    print("Starting Optimization")
    print("=" * 60)

    start_time = time.time()

    result = run_two_models(
        model1=model_mistral,
        model2=model_qwen,
        tokenizer1=tokenizer_mistral,
        tokenizer2=tokenizer_qwen,
        messages=multi_messages,
        target=TARGET,
        config=config
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total Time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg Time/Step: {total_time/config.num_steps:.2f}s")
    print(f"Best Loss:     {result.best_loss:.4f}")
    print(f"Best String:   {result.best_string}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Best String: {result.best_string}\n")
        f.write(f"Best Loss: {result.best_loss}\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"Avg Time/Step: {total_time/config.num_steps:.2f}s\n")
        f.write(f"\n--- Loss History ---\n")
        for i, (loss, string) in enumerate(zip(result.losses, result.strings)):
            if i % 100 == 0:
                f.write(f"Step {i}: loss={loss:.4f}, string={string[:50]}...\n")

    print(f"\nResults saved to: {OUTPUT_PATH}")

    verify_results(
        result.best_string,
        multi_messages,
        model_qwen, tokenizer_qwen,
        model_mistral, tokenizer_mistral
    )

    return result


if __name__ == "__main__":
    main()
