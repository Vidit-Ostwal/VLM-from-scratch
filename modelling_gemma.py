import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modelling_siglip import SigLipVisionConfig, SigLipVisionModel



class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]



class GemmaConfig():
    def __init__(
        self,
        vocab_size,                    # Total number of tokens in the vocabulary
        hidden_size,                   # Dimension of the hidden states (used across the model)
        intermediate_size,             # Size of the feedforward layer in transformer blocks
        num_hidden_layers,             # Total number of transformer layers (depth of the model)
        num_attention_heads,           # Number of attention heads in multi-head self-attention
        num_key_value_heads,           # Number of heads used for key/value caching in attention (for KV-sharing or Flash attention)
        head_dim=256,                  # Dimensionality of each attention head
        max_position_embeddings=8192,  # Maximum number of positional embeddings the model can handle
        rms_norm_eps=1e-6,             # Epsilon value for RMSNorm to avoid divide-by-zero
        rope_theta=10000.0,            # Base frequency used in RoPE (Rotary Positional Embeddings)
        attention_bias=False,          # Whether to use bias terms in attention projections
        attention_dropout=0.0,         # Dropout applied in attention (usually 0 during inference)
        pad_token_id=None,             # Token ID used for padding in input sequences
        **kwargs,                      # Additional unused keyword arguments (helps compatibility)
    ):
        super().__init__()  # Call the base class (optional here, unless you're subclassing)

        # Assign parameters to class attributes
        self.vocab_size = vocab_size                                # Number of tokens the model understands
        self.max_position_embeddings = max_position_embeddings      # How long an input sequence can be (in tokens)
        self.hidden_size = hidden_size                              # Core dimensionality for most layers
        self.intermediate_size = intermediate_size                  # Size of the MLP layer in each transformer block
        self.num_hidden_layers = num_hidden_layers                  # Depth of the transformer stack
        self.num_attention_heads = num_attention_heads              # Number of heads in attention layers
        self.head_dim = head_dim                                    # Dim of each head (head_dim * num_heads = hidden_size)
        self.num_key_value_heads = num_key_value_heads              # Heads used for KV caching or optimization
        self.rms_norm_eps = rms_norm_eps                            # Stability constant for RMSNorm
        self.rope_theta = rope_theta                                # Controls frequency range in rotary positional embeddings
        self.attention_bias = attention_bias                        # Toggle to include bias in attention layers
        self.attention_dropout = attention_dropout                  # Dropout probability in attention (0.0 = no dropout)
        self.pad_token_id = pad_token_id                            # Padding token ID (used to mask out attention)


class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,              # Dict containing config params for the vision encoder (e.g., image size, patch size, etc.)
        text_config=None,                # Dict containing config params for the language model (e.g., hidden size, vocab size, etc.)
        ignore_index=-100,              # Used during training to ignore loss computation at specific token positions
        image_token_index=256000,       # Token ID used to represent image tokens in the input sequence
        vocab_size=257152,              # Total number of tokens in the vocabulary (includes text + special tokens like image tokens)
        projection_dim=2048,            # Dimensionality of projected vision embeddings (before feeding into LM)
        hidden_size=2048,               # Hidden size of the language model (used for projection or alignment)
        pad_token_id=None,              # ID of the padding token (used if padding is required, e.g., during batching)
        **kwargs,                       # Catch-all for additional parameters passed to the superclass or for future use
    ):
        super().__init__()              # Call the base class constructor

        # Save basic parameters for later use
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False  # Indicates this is a decoder-only model (like GPT-style), not encoder-decoder

        self.pad_token_id = pad_token_id

        # Create vision configuration object (e.g., image_size, patch_size, etc.)
        self.vision_config = SigLipVisionConfig(**vision_config)

        # Store raw text config dict (optional, not used directly)
        self.text_config = text_config

        # Create text configuration object (e.g., hidden size, vocab size, etc.)
        # Also inject pad_token_id into it
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

        # Update vocab size to match what GemmaConfig says (may override earlier value)
        self.vocab_size = self.text_config.vocab_size

        # Compute how many image tokens the vision encoder will produce
        # e.g., for 224x224 image with 14x14 patches → (224/14)^2 = 256 tokens
        self.text_config.num_image_tokens = (
            (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        )

        # Also store projection dimension into vision config for consistency
        self.vision_config.projection_dim = projection_dim



def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size                       # Hidden size of input/output
        self.intermediate_size = config.intermediate_size           # Size of intermediate projection

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # [B, L, H] -> [B, L, I]
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)    # [B, L, H] -> [B, L, I]
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # [B, L, I] -> [B, L, H]

    def forward(self, x):  # x: [Batch_Size, Seq_Len, Hidden_Size]
        gated = nn.functional.gelu(self.gate_proj(x), approximate="tanh")  # [B, L, I]
        up = self.up_proj(x)                                               # [B, L, I]
        fused = gated * up                                                 # [B, L, I]
        output = self.down_proj(fused)                                     # [B, L, H]
        return output  # [Batch_Size, Seq_Len, Hidden_Size]


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma), initialized to 0
        # Shape: [dim]
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # x: [Batch_Size, Seq_Len, Hidden_Size]
        # Compute root mean square normalization across the last dimension (Hidden_Size)
        # Output: same shape as x -> [Batch_Size, Seq_Len, Hidden_Size]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x: [Batch_Size, Seq_Len, Hidden_Size]
        # Step 1: Convert to float32 for stability during norm computation
        output = self._norm(x.float())  # [Batch_Size, Seq_Len, Hidden_Size]

        # Step 2: Scale normalized output using (1 + weight)
        # Unlike LLaMA which uses x.to(float16) * w, Gemma does (x * w).to(float16)
        output = output * (1.0 + self.weight.float())  # [Batch_Size, Seq_Len, Hidden_Size]

        # Step 3: Convert back to original input dtype (e.g., float16 or bfloat16)
        return output.type_as(x)  # [Batch_Size, Seq_Len, Hidden_Size]


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim  # head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))  # [Head_Dim // 2]
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)  # [Head_Dim // 2]

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):  # x: [B, H, L, D]
        self.inv_freq.to(x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # [B, D//2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, L]

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, L, D//2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, L, D]
            cos = emb.cos()  # [B, L, D]
            sin = emb.sin()  # [B, L, D]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)  # [B, L, D], [B, L, D]


class GemmaAttention(nn.Module): 

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads              # Number of query heads
        self.head_dim = config.head_dim                          # Dim per head
        self.num_key_value_heads = config.num_key_value_heads    # May be smaller than num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # Causal attention

        assert self.hidden_size % self.num_heads == 0

        # Projections for Q, K, V, and output
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Rotary position embedding module
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,                           # [Batch_Size, Seq_Len, Hidden_Size]
        attention_mask: Optional[torch.Tensor] = None,         # [Batch_Size, 1, Seq_Len_Q, Seq_Len_KV]
        position_ids: Optional[torch.LongTensor] = None,       # [Batch_Size, Seq_Len]
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.size()  # [B, L, H]

        # Linear projections from hidden to Q/K/V
        # [Batch_Size, Seq_Len, Num_Heads * Head_Dim]
        query_states = self.q_proj(hidden_states)              # [B, L, num_heads * head_dim]
        key_states   = self.k_proj(hidden_states)              # [B, L, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)              # [B, L, num_kv_heads * head_dim]

        # Reshape and transpose to: [Batch_Size, Num_Heads, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Compute rotary embeddings
        # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        # Apply rotary positional encoding to Q and K
        # query_states: [Batch_Size, Num_Heads, Seq_Len, Head_Dim]
        # key_states:   [Batch_Size, Num_KV_Heads, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Update KV cache if available (in generation mode)
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat KV heads to match Q heads
        # key_states, value_states: [Batch_Size, Num_Heads, Seq_Len, Head_Dim]
        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention: [B, H, L_Q, L_K]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))  # [B, H, L, L]
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Apply causal attention mask
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask  # [B, H, L_Q, L_KV]

        # Apply softmax and attention dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [B, H, L, L]
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute attention output
        # [B, H, L_Q, L_KV] x [B, H, L_KV, D] → [B, H, L_Q, D]
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate expected shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        # Merge heads back to [B, L, H]
        attn_output = attn_output.transpose(1, 2).contiguous()                  # [B, L, H, D]
        attn_output = attn_output.view(bsz, q_len, -1)                          # [B, L, H*D]

        # Final projection to hidden_size
        attn_output = self.o_proj(attn_output)                                 # [B, L, Hidden_Size]

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Multi-head self-attention module with rotary embeddings and optional KV cache
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        # Feed-forward network (MLP)
        self.mlp = GemmaMLP(config)

        # LayerNorm applied before attention
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LayerNorm applied after attention and before MLP
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,                          # [Batch_Size, Seq_Len, Hidden_Size]
        attention_mask: Optional[torch.Tensor] = None,        # [Batch_Size, Seq_Len]
        position_ids: Optional[torch.LongTensor] = None,      # [Batch_Size, Seq_Len]
        kv_cache: Optional[KVCache] = None,                 # Cached keys and values for fast decoding
    ) -> torch.FloatTensor:
        
        # Save residual for attention block
        residual = hidden_states  # [Batch_Size, Seq_Len, Hidden_Size]

        # Apply pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)  # [Batch_Size, Seq_Len, Hidden_Size]

        # Apply self-attention (with optional rotary position and KV caching)
        # Output: hidden_states [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # Add residual connection after attention
        hidden_states = residual + hidden_states  # [Batch_Size, Seq_Len, Hidden_Size]

        # Save residual for MLP block
        residual = hidden_states  # [Batch_Size, Seq_Len, Hidden_Size]

        # Apply post-attention normalization
        hidden_states = self.post_attention_layernorm(hidden_states)  # [Batch_Size, Seq_Len, Hidden_Size]

        # Apply feed-forward network
        hidden_states = self.mlp(hidden_states)  # [Batch_Size, Seq_Len, Hidden_Size]

        # Add residual connection after MLP
        hidden_states = residual + hidden_states  # [Batch_Size, Seq_Len, Hidden_Size]

        return hidden_states  # Final output of this decoder layer



class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding layer
        # Shape: [Vocab_Size, Hidden_Size]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final RMS normalization applied after all layers
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        # Returns embedding matrix [Vocab_Size, Hidden_Size]
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,       # [Batch_Size, Seq_Len]
        position_ids: Optional[torch.LongTensor] = None,     # [Batch_Size, Seq_Len]
        inputs_embeds: Optional[torch.FloatTensor] = None,   # [Batch_Size, Seq_Len, Hidden_Size]
        kv_cache: Optional[KVCache] = None,                # Cache for key/value tensors (used in generation)
    ) -> torch.FloatTensor:

        # Start with input embeddings
        # hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds

        # Normalize embeddings by sqrt(hidden_size), standard practice in transformers
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer  # Still [Batch_Size, Seq_Len, Hidden_Size]

        # Apply each decoder layer in sequence
        for decoder_layer in self.layers:
            # Each layer returns: [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # Final layer normalization
        # hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        return hidden_states  # Final output of the transformer model


class GemmaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)  # Backbone model for computing hidden states
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # Final projection to vocabulary logits

    def get_input_embeddings(self):
        # Returns the input embedding layer
        return self.model.embed_tokens
    
    def tie_weights(self):
        # Tie the weights of the final LM head to the input embeddings
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,       # [Batch_Size, Seq_Len]
        position_ids: Optional[torch.LongTensor] = None,     # [Batch_Size, Seq_Len]
        inputs_embeds: Optional[torch.FloatTensor] = None,   # [Batch_Size, Seq_Len, Hidden_Size]
        kv_cache: Optional[KVCache] = None,                # Optional key-value cache for efficient decoding
    ) -> Tuple:

        # Pass inputs through the base model
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs  # [Batch_Size, Seq_Len, Hidden_Size]

        # Project hidden states to vocabulary logits
        # logits: [Batch_Size, Seq_Len, Vocab_Size]
        logits = self.lm_head(hidden_states)

        # Convert logits to float32 for numerical stability (important during mixed-precision training)
        logits = logits.float()

        # Prepare return dictionary
        return_data = {
            "logits": logits,  # [Batch_Size, Seq_Len, Vocab_Size]
        }

        # Include kv_cache in output if used (for incremental generation)
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear projection from vision encoder hidden size to projection dimension
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True
        )

    def forward(self, image_features):
        # [batch_size, num_patches, hidden_size] -> [batch_size, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model =  GemmaForCasualLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,     # Shape: (batch_size, num_patches, hidden_dim)
        inputs_embeds: torch.Tensor,      # Shape: (batch_size, seq_len, hidden_dim)
        input_ids: torch.Tensor,          # Shape: (batch_size, seq_len)
        attention_mask: torch.Tensor,     # Shape: (batch_size, seq_len)
        kv_cache: Optional[KVCache] = None
    ):

        _, _, embed_dim = image_features.shape                                      # Extract embedding dimension from image features (B, num_patches, hidden_dim)
        batch_size, sequence_length = input_ids.shape                               # Get batch size and sequence length from input token IDs (B, seq_len)
        dtype, device = inputs_embeds.dtype, inputs_embeds.device                   # Store dtype and device info for consistency
        scaled_image_features = image_features / (self.config.hidden_size**0.5)     # Scale image features by sqrt(hidden_size) for normalization

    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)  # Shape: (batch_size, seq_len, embed_dim)
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)  # Shape: (batch_size, seq_len). True for text tokens
        image_mask = input_ids == self.config.image_token_index                                      # Shape: (batch_size, seq_len). True for image tokens
        pad_mask = input_ids == self.pad_token_id                                                    # Shape: (batch_size, seq_len). True for padding tokens

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)     # Shape: (batch_size, seq_len, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)       # Shape: (batch_size, seq_len, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)   # Shape: (batch_size, seq_len, embed_dim)


        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)  # Fill in text token positions

        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)  # Replace image token positions

        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)  # Clear padding positions


        #### CREATE THE ATTENTION MASK ####>
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
        # final_embedding: [batch_size, q_len, hidden_dim]         → embeddings for the current input tokens
        # causal_mask:     [batch_size, 1, q_len, kv_len]          → attention mask broadcasted across heads
        # position_ids:    [batch_size, q_len]                     → position index for each query token


    def forward(
        self,
        input_ids: torch.LongTensor = None,                     # Token IDs of the input text
        pixel_values: torch.FloatTensor = None,                 # Raw image tensors (usually shape [B, 3, H, W])
        attention_mask: Optional[torch.Tensor] = None,          # Attention mask indicating valid tokens (1 = keep, 0 = pad)
        kv_cache: Optional[KVCache] = None,                     # Key/Value cache for efficient generation (used in decoder-only transformers)
    ) -> Tuple:

        # Ensure no padding in the input, since right-padded input isn't supported
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # --- Step 1: Get text embeddings ---
        # Converts input token IDs into embeddings using the LM's embedding layer
        # Shape: (batch_size, seq_len, hidden_dim)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # --- Step 2: Get image features ---
        # Pass raw image pixels through vision encoder to get patch embeddings
        # Output shape: (batch_size, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # Project vision encoder output to match LM hidden size
        # Shape: (batch_size, num_patches, hidden_dim)
        image_features = self.multi_modal_projector(selected_image_feature)

        # --- Step 3: Combine text and image features ---
        # Merge image embeddings with text embeddings
        # Also update attention mask and generate position_ids if needed
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features,      # vision embeddings
            inputs_embeds,       # text embeddings
            input_ids,           # raw input token IDs
            attention_mask,      # original attention mask
            kv_cache             # optional cache for decoding
        )

        # --- Step 4: Pass through the language model ---
        # Feed combined embeddings into the LM
        outputs = self.language_model(
            attention_mask=attention_mask,     # updated attention mask
            position_ids=position_ids,         # position IDs for combined sequence
            inputs_embeds=inputs_embeds,       # combined embeddings
            kv_cache=kv_cache,                 # optional cache for generation
        )

        return outputs  # Model output (e.g., logits, hidden states, etc.)
