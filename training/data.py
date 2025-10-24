"""
Data loading utilities for H-Net training.

This module provides:
- PackedDataset: Dataset that loads and packs sequences
- PackedDataCollator: Collator for creating packed batches with cu_seqlens
- HuggingFaceStreamingDataset: Streaming dataset for HuggingFace datasets
"""

import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from hnet.utils.tokenizers import ByteTokenizer


class PackedDataset(Dataset):
    """
    Dataset for loading text data and packing sequences efficiently.

    This dataset handles:
    - Loading text data from files
    - Tokenization using ByteTokenizer
    - Creating packed sequences to maximize GPU utilization

    Note: For large datasets (100B tokens), use StreamingDataset instead.
    """

    def __init__(
        self,
        data_path: str,
        max_seq_length: int = 2048,
        tokenizer: Optional[ByteTokenizer] = None,
        add_bos: bool = True,
    ):
        """
        Initialize the packed dataset.

        Args:
            data_path: Path to the data file(s)
            max_seq_length: Maximum sequence length
            tokenizer: ByteTokenizer instance (creates one if None)
            add_bos: Whether to add BOS token
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
        self.add_bos = add_bos

        # Load data
        self.documents = self._load_data(data_path)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.

        Returns:
            Dictionary with 'input_ids' and other metadata
        """
        text = self.documents[idx]
        input_ids = self._tokenize_text(text)

        return {"input_ids": input_ids}

    def _load_data(self, data_path: str) -> List[str]:
        """Load data from file(s)."""
        documents = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(line)

        return documents

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize a text string."""
        encoded = self.tokenizer.encode([text], add_bos=self.add_bos)[0]
        input_ids = encoded["input_ids"]

        # Truncate if needed
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]

        return input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)


class PackedDataCollator:
    """
    Collator for creating packed batches with cumulative sequence lengths.

    This collator:
    - Packs multiple sequences into a single batch
    - Computes cu_seqlens for flash-attention
    - Handles variable-length sequences efficiently
    """

    def __init__(
        self,
        max_seq_length: int = 2048,
        return_mask: bool = False,
    ):
        """
        Initialize the collator.

        Args:
            max_seq_length: Maximum total sequence length in batch
            return_mask: Whether to return attention masks
        """
        self.max_seq_length = max_seq_length
        self.return_mask = return_mask

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a packed batch.

        Args:
            examples: List of examples from dataset

        Returns:
            Dictionary containing:
            - input_ids: Packed input tokens (B*L,) or (B, L)
            - cu_seqlens: Cumulative sequence lengths (B+1,)
            - max_seqlen: Maximum sequence length in batch
            - mask: Optional attention mask
        """
        # Extract input_ids from examples (can be numpy arrays or lists)
        sequences = []
        for ex in examples:
            if isinstance(ex, dict):
                ids = ex["input_ids"]
            else:
                ids = ex

            # Convert to tensor if needed
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)

            # Truncate if needed
            if len(ids) > self.max_seq_length:
                ids = ids[: self.max_seq_length]

            sequences.append(ids)

        # Pack sequences
        result = self._pack_sequences(sequences)

        return result

    def _pack_sequences(self, sequences: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pack sequences into a single tensor with cu_seqlens."""
        # Compute sequence lengths
        seq_lens = [len(seq) for seq in sequences]

        # Compute cumulative sequence lengths
        cu_seqlens = torch.tensor([0] + seq_lens, dtype=torch.int32).cumsum(0)

        # For causal LM, we need to shift targets by 1 position
        # Input:  [tok0, tok1, tok2, tok3]
        # Target: [tok1, tok2, tok3, tok4]
        #
        # In packed mode with multiple sequences, we shift within each sequence:
        # Seq1: [a, b, c] Seq2: [x, y, z]
        # Input:  [a, b, c, x, y, z]
        # Target: [b, c, ?, y, z, ?]  where ? is padding/ignore

        input_sequences = []
        target_sequences = []

        for seq in sequences:
            if len(seq) > 1:
                # Input is all tokens except the last
                input_sequences.append(seq[:-1])
                # Target is all tokens except the first
                target_sequences.append(seq[1:])
            else:
                # Skip sequences that are too short (can't create input/target pair)
                continue

        # Pack sequences
        input_ids = (
            torch.cat(input_sequences, dim=0)
            if input_sequences
            else torch.tensor([], dtype=torch.long)
        )
        targets = (
            torch.cat(target_sequences, dim=0)
            if target_sequences
            else torch.tensor([], dtype=torch.long)
        )

        # Recompute sequence lengths after shift (each sequence is now 1 token shorter)
        seq_lens = [len(seq) - 1 for seq in sequences if len(seq) > 1]
        cu_seqlens = torch.tensor([0] + seq_lens, dtype=torch.int32).cumsum(0)

        # Compute max sequence length
        max_seqlen = max(seq_lens) if seq_lens else 0

        result = {
            "input_ids": input_ids,
            "targets": targets,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }

        if self.return_mask:
            # Create a simple mask (not typically needed for packed mode)
            # This would be used for non-packed mode
            pass

        return result


class TextDataset(Dataset):
    """
    Simple text dataset that loads documents from a file.

    Each line in the file is treated as a separate document.
    Map-style dataset that loads all data into memory.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[ByteTokenizer] = None,
        max_length: Optional[int] = None,
        add_bos: bool = True,
    ):
        """
        Initialize text dataset.

        Args:
            data_path: Path to text file (one document per line)
            tokenizer: ByteTokenizer instance
            max_length: Maximum length to truncate documents
            add_bos: Whether to add BOS token
        """
        self.data_path = data_path
        self.tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
        self.max_length = max_length
        self.add_bos = add_bos

        # Load all documents
        self.documents = self._load_documents(data_path)

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a document by index."""
        text = self.documents[idx]

        # Tokenize
        encoded = self.tokenizer.encode([text], add_bos=self.add_bos)[0]
        input_ids = encoded["input_ids"]

        # Truncate if needed
        if self.max_length is not None and len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        return {"input_ids": input_ids}

    def _load_documents(self, data_path: str) -> List[str]:
        """Load all documents from file."""
        documents = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    documents.append(line)

        return documents


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for large-scale training.

    Loads data on-the-fly without loading entire dataset into memory.
    Iterable-style dataset for streaming large text files.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[ByteTokenizer] = None,
        chunk_size: int = 2048,
        buffer_size: int = 10000,
        add_bos: bool = True,
    ):
        """
        Initialize streaming dataset.

        Args:
            data_path: Path to data file(s) or glob pattern
            tokenizer: ByteTokenizer instance
            chunk_size: Target chunk size for reading
            buffer_size: Number of documents to buffer
            add_bos: Whether to add BOS token
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.add_bos = add_bos

        # Handle glob patterns
        if "*" in data_path:
            import glob

            self.file_paths = sorted(glob.glob(data_path))
        else:
            self.file_paths = [data_path] if os.path.isfile(data_path) else []

            # If it's a directory, get all .txt files
            if os.path.isdir(data_path):
                self.file_paths = sorted(
                    [
                        os.path.join(data_path, f)
                        for f in os.listdir(data_path)
                        if f.endswith(".txt")
                    ]
                )

        if not self.file_paths:
            raise ValueError(f"No files found at {data_path}")

    def __iter__(self):
        """Iterate over the dataset."""
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Tokenize the document
                    encoded = self.tokenizer.encode([line], add_bos=self.add_bos)[0]
                    input_ids = encoded["input_ids"]

                    yield {"input_ids": input_ids}

    def _read_chunks(self):
        """Read data in chunks."""
        # This is a simplified version - for production, you might want
        # to implement more sophisticated chunking
        return iter(self)


class HuggingFaceStreamingDataset(IterableDataset):
    """
    Streaming dataset for HuggingFace datasets.

    Designed for large-scale datasets like FineWeb-Edu (100B tokens).
    Inherits from IterableDataset to work properly with PyTorch DataLoader.
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer: Optional[ByteTokenizer] = None,
        text_column: str = "text",
        add_bos: bool = True,
        max_length: Optional[int] = None,
    ):
        """
        Initialize HuggingFace streaming dataset.

        Args:
            hf_dataset: HuggingFace dataset (can be streaming)
            tokenizer: ByteTokenizer instance
            text_column: Name of the text column in the dataset
            add_bos: Whether to add BOS token
            max_length: Maximum length to truncate documents
        """
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
        self.text_column = text_column
        self.add_bos = add_bos
        self.max_length = max_length

    def __iter__(self):
        """Iterate over the dataset."""
        for example in self.hf_dataset:
            # Get text from the example
            text = example[self.text_column]

            if not text or not text.strip():
                continue

            # Tokenize the document
            encoded = self.tokenizer.encode([text], add_bos=self.add_bos)[0]
            input_ids = encoded["input_ids"]

            # Truncate if needed
            if self.max_length is not None and len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]

            yield {"input_ids": input_ids}
