import numpy as np
import pandas as pd
import torch

from conftest import load_script_module


decoder_ablation = load_script_module("analyze_decoder_action_token_dependence.py")


class _DummyModel:
    @property
    def action_shape(self) -> tuple[int, int]:
        return (2, 2)


def test_reshape_action_tokens_uses_model_action_shape():
    model = _DummyModel()
    tokens = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    action_tokens = decoder_ablation.reshape_action_tokens(model, tokens)

    assert tuple(action_tokens.shape) == (2, 1, 2, 2, 3)
    assert torch.equal(action_tokens[0, 0, 0, 0], tokens[0, 0])
    assert torch.equal(action_tokens[0, 0, 1, 1], tokens[0, 3])


def test_summarize_loss_table_returns_condition_and_comparison_tables():
    loss_df = pd.DataFrame(
        {
            "index": [0, 1],
            "loss_normal": [1.0, 2.0],
            "loss_hard_codebook": [1.5, 2.5],
            "loss_shuffled_action_tokens": [2.0, 4.0],
            "loss_zeroed_action_tokens": [3.0, 3.0],
        }
    )

    summary_df, comparison_df = decoder_ablation.summarize_loss_table(loss_df)

    assert set(summary_df["condition"]) == {
        "normal",
        "hard_codebook",
        "shuffled_action_tokens",
        "zeroed_action_tokens",
    }
    normal_row = summary_df[summary_df["condition"] == "normal"].iloc[0]
    assert abs(float(normal_row["mean"]) - 1.5) < 1e-6

    shuffled_row = comparison_df[comparison_df["condition"] == "shuffled_action_tokens"].iloc[0]
    assert abs(float(shuffled_row["mean_delta_vs_normal"]) - 1.5) < 1e-6
    assert abs(float(shuffled_row["mean_ratio_vs_normal"]) - 2.0) < 1e-6
    assert abs(float(shuffled_row["fraction_worse_than_normal"]) - 1.0) < 1e-6


def test_compute_quantized_token_variants_preserves_batch_and_code_shape():
    class _DummyVQ:
        eps = 1e-12

        def __init__(self):
            self.embedding_dim = 2
            self.codebooks = torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )

        def encode(self, input_data: torch.Tensor, batch_size: int) -> torch.Tensor:
            return input_data.reshape(batch_size, -1, input_data.shape[-1]).reshape(-1, input_data.shape[-1])

        def decode(self, quantized_input: torch.Tensor, batch_size: int) -> torch.Tensor:
            return quantized_input.reshape(batch_size, -1, quantized_input.shape[-1])

    class _DummyQuantizedModel:
        def __init__(self):
            self.vq = _DummyVQ()

    torch.manual_seed(0)
    model = _DummyQuantizedModel()
    first_tokens_flat = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    last_tokens_flat = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    relaxed_tokens, hard_tokens, indices = decoder_ablation.compute_quantized_token_variants(
        model,
        first_tokens_flat,
        last_tokens_flat,
    )

    assert tuple(relaxed_tokens.shape) == (2, 2, 2)
    assert tuple(hard_tokens.shape) == (2, 2, 2)
    assert tuple(indices.shape) == (2, 2)
    assert np.array_equal(indices.detach().cpu().numpy(), np.array([[1, 2], [1, 1]], dtype=np.int64))
