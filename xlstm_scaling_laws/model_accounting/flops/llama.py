from .attention_op import (
    count_flops_attention_generation,
    count_flops_attention_prefill,
)
from .llama_backbone import (
    count_flops_llama_backbone,
)


def count_llama_model_flops__prefill(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    d_qk,
    d_hv,
    n_headkv,
    seq_len,
    with_unembed=False,  # whether to count the final linear layer
    return_attenion_op_flops=False,
    **kwargs,
):
    """Count the flops for the LLaMA model in prefill mode for a batch of sequences."""

    flops_backbone = count_flops_llama_backbone(
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        n_headkv=n_headkv,
        with_unembed=with_unembed,
    )

    flops_attention = count_flops_attention_prefill(
        seq_len=seq_len,
        d_qk=d_qk,
        d_hv=d_hv,
        n_headq=n_headq,
    )

    total_flops = seq_len * flops_backbone + n_blocks * flops_attention

    # during inference we actually compute the output logits
    # for the very last token in the sequence,
    # we add these flops in case with_unembed is False
    if not with_unembed:
        total_flops += 2 * n_headq * d_hv * n_vocab

    total_flops *= float(batch_size)
    if return_attenion_op_flops:
        return total_flops, n_blocks * flops_attention
    else:
        return total_flops


def count_llama_model_flops__generation(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    d_qk,
    n_headkv,
    d_hv,
    seq_len_pre,
    seq_len_gen,
    **kwargs,
):
    """Count the flops for the LLaMA model in generation mode for a batch of sequences."""

    flops_backbone = count_flops_llama_backbone(
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        n_headkv=n_headkv,
        with_unembed=False, # we do not compute the unembedding for prefill, only during generation
    )

    flops_attention = count_flops_attention_generation(
        seq_len_pre=seq_len_pre,
        seq_len_gen=seq_len_gen,
        d_qk=d_qk,
        d_hv=d_hv,
        n_headq=n_headq,
    )

    total_flops = batch_size * (
        (seq_len_pre + seq_len_gen) * flops_backbone + n_blocks * flops_attention
    )

    # during inference we actually compute the output logits
    # for every token in the generation phase,
    # we add these flops
    total_flops += batch_size * 2 * n_headq * d_qk * n_vocab * (seq_len_gen + 1)

    return total_flops
