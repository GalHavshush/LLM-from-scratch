import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#################### 01 - Working with text data ####################

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

#####################################################################



#################### 02 - Coding attention mechanisms ###############

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
    
#####################################################################    
    

#################### 03 - Implementing a GPT Model ##################

class LayerNorm(nn.Module):
    """
    Simple implementation of Layer Normalization.

    This normalizes each input vector across its last dimension
    (typically the embedding dimension) to have mean ~0 and variance ~1,
    then applies learnable scale and shift parameters.
    """
    def __init__(self, emb_dim):
        """
        Initialize LayerNorm.

        Args:
            emb_dim (int): Size of the embedding dimension to normalize.
        """
        super().__init__()
        self.eps = 1e-5  # Small constant for numerical stability
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Learnable scaling parameter (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # Learnable shifting parameter (beta)

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x (Tensor): Input tensor (..., emb_dim).

        Returns:
            Tensor: Normalized tensor with same shape as input.
        """
        
        mean = x.mean(dim=-1, keepdim=True) # Compute mean across the last dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False) # Compute variance across the last dimension (unbiased=False matches standard LayerNorm behavior)
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # Normalize input (zero mean, unit variance)
        return self.scale * norm_x + self.shift # Apply learnable scale and shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Apply GELU activation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Activated tensor with same shape as input.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Feedforward network used inside Transformer blocks.

    Typical Transformer MLP structure:
      Linear expansion -> GELU activation -> Linear projection back.
    Usually expands embedding dimension by ~4x before projecting back.
    """
    def __init__(self, cfg):
        """
        Initialize feedforward network.

        Args:
            cfg (dict): Model configuration containing "emb_dim".
        """
        super().__init__()

        # Two-layer MLP with GELU activation in between
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expand embedding dimension
            GELU(),                                         # Nonlinear activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Project back to original size
        )

    def forward(self, x):
        """
        Forward pass through feedforward network.

        Args:
            x (Tensor): Input tensor (batch, seq_len, emb_dim).

        Returns:
            Tensor: Output tensor with same shape as input.
        """
        return self.layers(x)
    

import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    A single Transformer block (pre-norm) consisting of:

      1) LayerNorm -> Multi-Head Causal Self-Attention -> Dropout -> Residual add
      2) LayerNorm -> FeedForward (MLP)               -> Dropout -> Residual add

    This is the standard building block used in GPT-style architectures.
    """
    def __init__(self, cfg):
        """
        Initialize a Transformer block.

        Args:
            cfg (dict): Configuration dictionary containing:
                - "emb_dim" (int): Embedding dimension.
                - "context_length" (int): Max sequence length for causal masking.
                - "n_heads" (int): Number of attention heads.
                - "drop_rate" (float): Dropout probability.
                - "qkv_bias" (bool): Whether to include bias in Q/K/V projections.
        """
        super().__init__()

        # Multi-head causal self-attention sublayer
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        
        self.ff = FeedForward(cfg) # Feedforward (MLP) sublayer

        # Layer norms used in pre-norm Transformer blocks
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.drop_shortcut = nn.Dropout(cfg["drop_rate"]) # Dropout applied on the residual branch outputs (after attn/ff)

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, emb_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_tokens, emb_dim).
        """
        
        shortcut = x # Save input for the first residual (shortcut) connection

        # Pre-norm, then attention
        x = self.norm1(x)
        x = self.att(x)

        x = self.drop_shortcut(x) # Dropout on attention output before adding residual
        x = x + shortcut # Residual connection: add original input back
        shortcut = x # Save for the second residual connection (post-attn representation)

        # Pre-norm, then feedforward (MLP)
        x = self.norm2(x)
        x = self.ff(x)

        
        x = self.drop_shortcut(x) # Dropout on feedforward output before adding residual        
        x = x + shortcut # Residual connection: add post-attn representation back

        return x
    

class GPTModel(nn.Module):
    """
    GPT-style Transformer language model.

    Architecture:
      - Token embeddings + positional embeddings
      - Dropout on embeddings
      - Stack of Transformer blocks (self-attention + MLP)
      - Final layer normalization
      - Linear output head producing logits over the vocabulary

    This model predicts the next token at each position
    (causal language modeling).
    """

    def __init__(self, cfg):
        """
        Initialize the GPT model.

        Args:
            cfg (dict): Configuration dictionary containing:
                - vocab_size (int): Vocabulary size.
                - context_length (int): Maximum sequence length.
                - emb_dim (int): Embedding dimension.
                - n_layers (int): Number of Transformer blocks.
                - drop_rate (float): Dropout probability.
        """
        super().__init__()

        
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # Token embedding layer: converts token IDs -> dense vectors
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # Positional embedding layer: encodes token position information
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # Dropout applied after summing token + positional embeddings
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) # Stack of Transformer blocks
        self.final_norm = LayerNorm(cfg["emb_dim"]) # Final normalization before output projection        
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # Output projection to vocabulary logits (no bias typical in GPT)

    def forward(self, in_idx):
        """
        Forward pass of the GPT model.

        Args:
            in_idx (Tensor): Input token IDs of shape (batch_size, seq_len).

        Returns:
            Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) # Convert token IDs into embeddings
        pos_ids = torch.arange(seq_len, device=in_idx.device) # Generate position indices on the correct device
        pos_embeds = self.pos_emb(pos_ids) # Look up positional embeddings
        x = tok_embeds + pos_embeds # Combine token and positional information
        x = self.drop_emb(x) # Apply embedding dropout
        x = self.trf_blocks(x) # Pass through stacked Transformer blocks
        x = self.final_norm(x) # Final layer normalization
        logits = self.out_head(x) # Project to vocabulary logits

        return logits
    
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text autoregressively using greedy decoding.

    At each step:
      - Feed the current token sequence to the model
      - Get logits for the next token
      - Convert logits to probabilities via softmax
      - Select the most probable token (argmax)
      - Append it to the sequence

    Args:
        model (nn.Module): Trained GPT-style model.
        idx (Tensor): Initial token indices (batch_size, seq_len).
        max_new_tokens (int): Number of tokens to generate.
        context_size (int): Maximum context window length.

    Returns:
        Tensor: Extended token sequence including generated tokens.
    """

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # Keep only the most recent tokens within context window
        with torch.no_grad(): # Disable gradient computation for inference
            logits = model(idx_cond)
        logits = logits[:, -1, :] # Take logits from last time step (next-token prediction)
        probas = torch.softmax(logits, dim=-1) # Convert logits to probabilities
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # Greedy decoding: pick highest-probability token
        idx = torch.cat((idx, idx_next), dim=1) # Append predicted token to the sequence

    return idx

#####################################################################    