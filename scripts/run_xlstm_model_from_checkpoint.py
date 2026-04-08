"""
This script demonstrates how to load an xLSTM model from a checkpoint and generate text with it. It first generates text with randomly initialized weights to show the difference, and then loads the pretrained weights and generates text again.
"""

import argparse

import torch
from transformers import AutoTokenizer

from xlstm_scaling_laws.checkpoint_loading import load_xlstm_model_from_checkpoint


def main(ckpt_path: str):

    torch.set_default_device("cuda:0")

    # this is a fork of EleutherAI/gpt-neox-20b
    tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

    text = "Tell me a joke. DO NOT REPEAT YOURSELF. Be concise."
    max_new_tokens = 128
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device="cuda:0")

    # Get the BOS token ID from the tokenizer
    bos_id = tokenizer.bos_token_id

    # Prepend BOS
    bos_tensor = torch.tensor([[bos_id]], device=tokens.device, dtype=tokens.dtype)
    tokens_with_bos = torch.cat([bos_tensor, tokens], dim=1)

    # Note: Feeding the tokens with out a leading bos_id token produces garbish outputs, because the model was trained with a bos token at the start of each sequence.
    # tokens_with_bos = tokens

    # Generate with random model
    print("Generating with random model...")
    xlstm = load_xlstm_model_from_checkpoint(ckpt_path, load_weights=False).to(
        device="cuda:0"
    )
    out = xlstm.generate(
        tokens_with_bos,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
    print(f"Prompt:\n{text}")
    print(f"Generated:\n{tokenizer.decode(out[0])}")
    print("\n" + "=" * 50 + "\n")

    # Generate with pretrained model
    print("\nGenerating with pretrained model...")
    xlstm = load_xlstm_model_from_checkpoint(ckpt_path, load_weights=True).to(
        device="cuda:0"
    )
    out = xlstm.generate(
        tokens_with_bos,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
    print(f"Prompt:\n{text}")
    print(f"Generated:\n{tokenizer.decode(out[0])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint to load.",
    )
    args = parser.parse_args()
    print(args)
    main(args.ckpt_path)
