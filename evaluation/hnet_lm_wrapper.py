"""
LM Evaluation Harness wrapper for H-Net models.

This module provides an adapter between byte-level H-Net models
and the token-based lm-eval harness API.
"""

from typing import List, Tuple

import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from evaluation.utils_eval import load_model_for_eval
from hnet.utils.tokenizers import ByteTokenizer


@register_model("hnet")
class HNetLM(LM):
    """
    Wrapper to make H-Net compatible with lm-eval harness.

    H-Net operates on bytes (vocab=256), while lm-eval expects
    token-level operations. This wrapper handles the conversion.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 2048,
    ):
        """
        Initialize H-Net LM wrapper.

        Args:
            model_path: Path to model checkpoint
            config_path: Path to model config JSON
            device: Device to use
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
        """
        super().__init__()

        self.model = load_model_for_eval(
            model_path, config_path, device=device, dtype=torch.bfloat16
        )
        self.tokenizer = ByteTokenizer()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def eot_token_id(self):
        """End of text token ID."""
        return self.tokenizer.eos_idx

    @property
    def max_length(self):
        """Maximum sequence length."""
        return self._max_length

    @property
    def max_gen_toks(self):
        """Maximum generation tokens."""
        return 256

    @property
    def batch_size(self):
        """Batch size."""
        return self._batch_size

    @property
    def device(self):
        """Device."""
        return self._device

    def tok_encode(self, text: str) -> List[int]:
        """
        Encode text to byte IDs.

        Args:
            text: Input text

        Returns:
            List of byte IDs
        """
        encoded = self.tokenizer.encode([text], add_bos=False)[0]
        return encoded["input_ids"].tolist()

    def tok_decode(self, tokens: List[int]) -> str:
        """
        Decode byte IDs to text.

        Args:
            tokens: List of byte IDs

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens, errors="replace")

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass on model.

        Args:
            input_ids: Input tensor [batch, seq_len]

        Returns:
            Logits tensor [batch, seq_len, vocab]
        """
        with torch.no_grad():
            output = self.model(input_ids)
            return output.logits

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log P(continuation | context) for each request.

        Args:
            requests: List of Instance objects with args=(context, continuation)

        Returns:
            List of (log_prob, is_greedy) tuples
        """
        from tqdm import tqdm

        results = []

        for request in tqdm(requests, desc="Computing loglikelihoods", disable=len(requests) < 10):
            # Extract context and continuation from Instance object
            context, continuation = request.args
            # Encode context and continuation
            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation)

            # Full sequence is context + continuation
            full_enc = context_enc + continuation_enc

            # Truncate if too long
            if len(full_enc) > self.max_length:
                # Try to keep as much continuation as possible
                context_len = len(context_enc)
                continuation_len = len(continuation_enc)
                max_context = self.max_length - continuation_len

                if max_context < 1:
                    # Continuation itself is too long
                    continuation_enc = continuation_enc[: self.max_length]
                    context_enc = []
                else:
                    # Truncate context from the left
                    context_enc = context_enc[-max_context:]

                full_enc = context_enc + continuation_enc

            # Convert to tensor
            input_ids = torch.tensor([full_enc], dtype=torch.long, device=self.device)

            # Get logits
            logits = self._model_call(input_ids)

            # Compute log likelihood for continuation
            # We need logits for positions that predict the continuation
            context_len = len(context_enc)
            continuation_len = len(continuation_enc)

            if continuation_len == 0:
                results.append((0.0, True))
                continue

            # Get logits for positions that predict continuation tokens
            # logits[i] predicts token[i+1]
            # So for continuation starting at position context_len,
            # we need logits from context_len-1 to context_len+continuation_len-2
            start_idx = max(0, context_len - 1)
            end_idx = context_len + continuation_len - 1

            pred_logits = logits[0, start_idx:end_idx, :]  # [cont_len, vocab]
            target_tokens = input_ids[0, context_len : context_len + continuation_len]

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[range(len(target_tokens)), target_tokens]  # [cont_len]

            # Sum log probs
            total_log_prob = token_log_probs.sum().item()

            # Check if greedy decoding would produce continuation
            greedy_tokens = pred_logits.argmax(dim=-1)
            is_greedy = torch.all(greedy_tokens == target_tokens).item()

            results.append((total_log_prob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests) -> List[Tuple[float,]]:
        """
        Compute rolling log likelihood for perplexity evaluation.

        Args:
            requests: List of Instance objects with args=(text,)

        Returns:
            List of (log_prob,) tuples
        """
        from tqdm import tqdm

        results = []

        for request in tqdm(
            requests, desc="Computing rolling loglikelihoods", disable=len(requests) < 10
        ):
            # Extract text from Instance object
            (text,) = request.args
            # Encode text
            tokens = self.tok_encode(text)

            if len(tokens) < 2:
                results.append((0.0,))
                continue

            # Truncate if necessary
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]

            # Convert to tensor
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Get logits
            logits = self._model_call(input_ids)

            # Compute log likelihood
            # logits[i] predicts token[i+1]
            pred_logits = logits[0, :-1, :]  # All but last position
            target_tokens = input_ids[0, 1:]  # All but first token

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[range(len(target_tokens)), target_tokens]

            # Sum log probs
            total_log_prob = token_log_probs.sum().item()

            results.append((total_log_prob,))

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate text until stop sequence.

        Note: This is optional for many zero-shot tasks.
        For now, we raise NotImplementedError.

        Args:
            requests: List of Instance objects

        Returns:
            List of generated strings
        """
        raise NotImplementedError(
            "Generation not implemented for H-Net wrapper. "
            "This is only needed for generative tasks."
        )
