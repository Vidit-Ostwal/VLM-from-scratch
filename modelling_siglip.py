from typing import Optional, Tuple
import torch
import torch.nn as nn

class SigLipVisionConfig:

    def __init__(
        self,
        hidden_size=786,                # Size of patch embeddings and all hidden states across the model, This defines the size of the embedding vector for each image patch after it's passed through the patch embedding layer.
        intermediate_size=3072,         # Hidden size of the feedforward (MLP) layer in each transformer block, typically 4x hidden_size, This is the size of the hidden layer in the Feedforward Neural Network (FFN) inside each Transformer block.
        num_hidden_layers=12,           # Number of transformer encoder layers in the model
        num_attention_heads=12,         # Number of self-attention heads in each attention layer
        num_channels=3,                 # Number of channels in the input image (3 for RGB images)
        image_size=224,                 # Height and width of the input image (assumes square images)
        patch_size=16,                  # Size of each image patch (image is divided into patches of patch_size x patch_size)
        layer_norm_eps=1e-6,            # Epsilon value to avoid division by zero in layer normalization
        attention_dropout=0.0,          # Dropout rate for attention probabilities
        num_image_tokens: int=None,     # Optional: Total number of image tokens after patching (can be inferred from image size and patch size)
        **kwargs                        # Additional keyword arguments (ignored or passed to super class if applicable)
    ):
        super().__init__()              # Call to base class constructor (can be removed if not subclassing from anything)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):  # Module to embed image patches + add positional encodings
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.config = config  # Store the full configuration for later use
        self.embed_dim = config.hidden_size  # Size of each patch embedding vector
        self.image_size = config.image_size  # Input image size (assumed square)
        self.patch_size = config.patch_size  # Size of each patch (assumed square)

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,  # Number of input image channels (e.g., 3 for RGB)
            out_channels=self.embed_dim,  # Output channels = embedding size for each patch
            kernel_size=self.patch_size,  # Each patch is patch_size x patch_size
            stride=self.patch_size,  # Move kernel with steps equal to patch size (non-overlapping)
            padding='valid'  # No zero padding; keeps patching cleanly aligned
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # Total number of patches in the image
        self.num_positions = self.num_patches  # One positional embedding per patch

        self.position_encodings = nn.Embedding(self.num_positions, self.embed_dim)  # Learnable positional embeddings of size [num_patches, embed_dim]

        self.register_buffer(
            "positions_ids",
            torch.arange(self.num_positions).expand((1, -1)),  # Create a tensor [0, 1, ..., num_patches-1] for indexing positional encodings
            persistent=False  # Buffer will not be saved in the model state_dict (used only during forward pass)
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape  # Input shape: [Batch_Size, Channels, Height, Width]

        # Convolve the image into non-overlapping patches using Conv2d
        # Output shape: [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embeddings(pixel_values)

        # Flatten the 2D spatial patches into a sequence: [Batch_Size, Embed_Dim, Num_Patches]
        patch_embeds = patch_embeds.flatten(2)

        # Transpose to match transformer input format: [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = patch_embeds.transpose(1, 2)

        # Add learnable positional embeddings to patch embeddings
        # position_encodings shape: [1, Num_Patches, Embed_Dim]
        embeddings = embeddings + self.position_encodings(self.positions_ids)

        return embeddings  # Final shape: [Batch_Size, Num_Patches, Embed_Dim]


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)      # Project to higher dimension (Intermediate_Size)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")  # Apply GELU activation (non-linear transform)
        hidden_states = self.fc2(hidden_states)      # Project back to original embedding dimension (Embed_Dim)
        return hidden_states                         # Return the transformed tensor


class SiglipAttention(nn.Module):  # Multi-headed attention from "Attention Is All You Need"
    def __init__(self, config):
        super().__init__()
        self.config = config                                          # Config with model hyperparameters
        self.embed_dim = config.hidden_size                          # Total embedding dimension (e.g., 768)
        self.num_heads = config.num_attention_heads                  # Number of attention heads (e.g., 12)
        self.head_dim = self.embed_dim // self.num_heads             # Dimension per head (e.g., 64)
        self.scale = self.head_dim**-0.5                              # Scaling factor for dot product attention
        self.dropout = config.attention_dropout                      # Dropout probability (e.g., 0.1)

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)      # Key projection: [E] -> [E]
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)      # Value projection: [E] -> [E]
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)      # Query projection: [E] -> [E]
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)    # Output projection: [E] -> [E]

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()                # hidden_states: [B, S, E]

        query_states = self.q_proj(hidden_states)                    # [B, S, E]
        key_states = self.k_proj(hidden_states)                      # [B, S, E]
        value_states = self.v_proj(hidden_states)                    # [B, S, E]

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, S, H, D] -> [B, H, S, D]
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)      # [B, S, H, D] -> [B, H, S, D]
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, S, H, D] -> [B, H, S, D]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale  # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):           # Check if attention shape is correct
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # Softmax over last dim -> [B, H, S, S]
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)              # Apply dropout to attention weights

        attn_output = torch.matmul(attn_weights, value_states)   # [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):  # Sanity check
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()      # [B, H, S, D] -> [B, S, H, D]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)  # [B, S, H * D] = [B, S, E]
        attn_output = self.out_proj(attn_output)                    # Final projection: [B, S, E] -> [B, S, E]

        return attn_output, attn_weights                            # Output: [B, S, E], attention weights: [B, H, S, S]


class SigLipEncoderLayer(nn.Module):
    def __init__(self, config : SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Normalizes across embed_dim with small epsilon for numerical stability
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # Normalizes across embed_dim with small epsilon for numerical stability


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states                     # Save input for first residual connection
        hidden_states = self.layer_norm1(hidden_states)  # Apply first LayerNorm (pre-norm before attention)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)  # Apply self-attention (ignore attention weights)
        hidden_states = residual + hidden_states     # Add skip connection (residual) after attention
        residual = hidden_states                     # Save new tensor for second residual connection
        hidden_states = self.layer_norm2(hidden_states)  # Apply second LayerNorm (pre-norm before MLP)
        hidden_states = self.mlp(hidden_states)      # Pass through MLP block (typically linear -> activation -> linear)
        hidden_states = residual + hidden_states     # Add skip connection (residual) after MLP
        return hidden_states                         # Return the output tensor


class SiglipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states  


class SigLipVisionTransformer(nn.Module):  # Define a Vision Transformer model for image inputs
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()  # Initialize nn.Module
        self.config = config  # Store configuration
        embed_dim = config.hidden_size  # Embedding dimension for each patch

        self.embeddings = SiglipVisionEmbeddings(config)  # Converts image to patch embeddings
        self.encoder = SiglipEncoder(config)  # Transformer encoder for processing patch embeddings
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # Final LayerNorm after encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values shape: [Batch, Channels, Height, Width]
        hidden_states = self.embeddings(pixel_values)  # Convert image into sequence of patch embeddings [Batch_Size, Num_Patches, Embed_Dim]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)  # Pass embeddings through transformer encoder
        last_hidden_state = self.post_layernorm(last_hidden_state)  # Normalize final hidden states
        return last_hidden_state  # Output shape: [Batch, Num_patches, Hidden_size]
    

class SigLipVisionModel:
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch Size, Channels, Heights, Width] -> [Batch_size, Num_patches, Embed_dim]
        return self.vision_model(pixel_values=pixel_values)
    