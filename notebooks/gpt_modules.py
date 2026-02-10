import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    A PyTorch Dataset for training a GPT-style language model using sliding windows
    over a tokenized text sequence.

    Given a long sequence of token IDs, this dataset:
    - Splits it into overlapping chunks of fixed length (max_length)
    - Uses the next-token prediction objective
    - Returns (input_ids, target_ids) pairs where targets are inputs shifted by one
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        # Lists to store input and target sequences
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire input text into a sequence of token IDs
        token_ids = tokenizer.encode(txt)

        # Create sliding windows over the token sequence
        # Each window produces:
        # - input_chunk:  tokens[i : i + max_length]
        # - target_chunk: tokens[i + 1 : i + max_length + 1]
        # This trains the model to predict the next token at each position
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Store as PyTorch tensors
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Return the number of training examples
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return a single (input_ids, target_ids) training pair
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=8,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    """
    Create a PyTorch DataLoader for training a GPT-style language model.

    This function:
    - Uses the GPT-2 tokenizer (via tiktoken)
    - Wraps the text in a GPTDatasetV1 using sliding windows
    - Returns a DataLoader that yields batches of (input_ids, target_ids)
    """

    # Initialize the GPT-2 BPE tokenizer from tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset using sliding windows over the tokenized text
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Wrap the dataset in a PyTorch DataLoader to enable batching and shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,   # Number of sequences per batch
        shuffle=shuffle,         # Whether to shuffle the dataset each epoch
        drop_last=drop_last,     # Drop the last batch if it is smaller than batch_size
        num_workers=num_workers  # Number of subprocesses used for data loading
    )

    return dataloader


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention using a single set of Q/K/V projections.

    This implementation:
      - Projects inputs into Q, K, V of size d_out
      - Splits into num_heads heads (each head has head_dim = d_out // num_heads)
      - Applies a causal mask so tokens cannot attend to future tokens
      - Applies dropout to attention weights
      - Concatenates heads and applies an output projection (out_proj)
    """
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize the multi-head attention layer.

        Args:
            d_in (int): Input embedding dimension.
            d_out (int): Output dimension (must be divisible by num_heads).
            context_length (int): Maximum sequence length for the causal mask.
            dropout (float): Dropout probability applied to attention weights.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to use bias in Q/K/V projection layers.
        """
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # Dimensionality per head (each head gets an equal slice of d_out)
        self.head_dim = d_out // num_heads

        # Single projection layers that produce concatenated Q/K/V for all heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Final output projection after concatenating all heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout applied to the attention weights
        self.dropout = nn.Dropout(dropout)

        # Register a causal mask (upper triangle) as a non-trainable buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_tokens, d_out).
        """
        # Unpack input shape
        b, num_tokens, d_in = x.shape

        # Project inputs to Q, K, V (still combined across heads in the last dim)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to split the last dimension into (num_heads, head_dim)
        # Result shapes: (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        # Transpose to move heads before tokens for efficient attention math
        # New shapes: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores per head
        # (b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        # -> (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Slice the causal mask to current sequence length and convert to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Apply causal mask: set scores for future positions to -inf before softmax
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Convert scores to probabilities with scaled softmax (scale by sqrt(d_k))
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values gives per-head context vectors
        # Result: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Ensure tensor is contiguous in memory before reshaping
        # Then merge (num_heads, head_dim) back into d_out
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        # Final linear projection mixes information across heads
        context_vec = self.out_proj(context_vec)

        return context_vec