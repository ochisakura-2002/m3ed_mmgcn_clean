"""
Collate functions for M3ED.

这个文件负责把多个 dialogue 样本拼成一个 batch。

由于每个 dialogue 的 utterance 数量不同，所以需要 padding。

输出 batch 的核心字段：
    text_features   : [batch_size, max_seq_len, text_dim]
    audio_features  : [batch_size, max_seq_len, audio_dim]
    visual_features : [batch_size, max_seq_len, visual_dim]
    labels          : [batch_size, max_seq_len]
    attention_mask  : [batch_size, max_seq_len]
    lengths         : [batch_size]
"""

from typing import Any, Dict, List

import torch


IGNORE_INDEX = -100


def m3ed_dialogue_collate_fn(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    M3ED dialogue-level collate function.

    参数：
        samples:
            一个 batch 内的 dialogue 样本列表。

    返回：
        padding 后的 batch 字典。
    """
    if len(samples) == 0:
        raise ValueError("Empty batch received by collate_fn.")

    batch_size = len(samples)
    lengths = torch.as_tensor(
        [sample["num_utterances"] for sample in samples],
        dtype=torch.long,
    )

    max_seq_len = int(lengths.max().item())

    text_dim = int(samples[0]["text_features"].shape[-1])
    audio_dim = int(samples[0]["audio_features"].shape[-1])
    visual_dim = int(samples[0]["visual_features"].shape[-1])

    text_features = torch.zeros(
        batch_size,
        max_seq_len,
        text_dim,
        dtype=torch.float32,
    )
    audio_features = torch.zeros(
        batch_size,
        max_seq_len,
        audio_dim,
        dtype=torch.float32,
    )
    visual_features = torch.zeros(
        batch_size,
        max_seq_len,
        visual_dim,
        dtype=torch.float32,
    )

    labels = torch.full(
        (batch_size, max_seq_len),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )

    speaker_ids_int = torch.full(
        (batch_size, max_seq_len),
        fill_value=-1,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        batch_size,
        max_seq_len,
        dtype=torch.bool,
    )

    for batch_index, sample in enumerate(samples):
        seq_len = sample["num_utterances"]

        text_features[batch_index, :seq_len] = sample["text_features"]
        audio_features[batch_index, :seq_len] = sample["audio_features"]
        visual_features[batch_index, :seq_len] = sample["visual_features"]

        labels[batch_index, :seq_len] = sample["labels"]
        speaker_ids_int[batch_index, :seq_len] = sample["speaker_ids_int"]

        attention_mask[batch_index, :seq_len] = True

    batch = {
        "text_features": text_features,
        "audio_features": audio_features,
        "visual_features": visual_features,
        "labels": labels,
        "speaker_ids_int": speaker_ids_int,
        "attention_mask": attention_mask,
        "lengths": lengths,

        "movie_ids": [sample["movie_id"] for sample in samples],
        "dialogue_ids": [sample["dialogue_id"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "utterance_ids": [sample["utterance_ids"] for sample in samples],
        "utterance_indices": [sample["utterance_indices"] for sample in samples],
        "texts": [sample["texts"] for sample in samples],
        "speaker_ids": [sample["speaker_ids"] for sample in samples],
        "label_names": [sample["label_names"] for sample in samples],

        "has_label_mismatch": [
            sample["has_label_mismatch"] for sample in samples
        ],
        "label_mismatch_positions": [
            sample["label_mismatch_positions"] for sample in samples
        ],
    }

    return batch