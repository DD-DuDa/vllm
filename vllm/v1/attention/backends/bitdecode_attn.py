# /home/ddy/Projects/vllm/vllm/v1/attention/backends/bitdecode_attn.py
# SPDX-License-Identifier: Apache-2.0

"""Attention layer with BitDecode quantized KV cache."""

from dataclasses import dataclass
import math
import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionType,
    MultipleOf,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# Import your custom kernels
from bit_decode import kvcache_pack_int, fwd_kvcache_int

logger = init_logger(__name__)


class BitDecodeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_quant_query_input: bool = False

    @staticmethod
    def get_name() -> str:
        return "BITDECODE"

    @staticmethod
    def get_impl_cls() -> type["BitDecodeAttentionImpl"]:
        return BitDecodeAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return BitDecodeAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["BitDecodeAttentionMetadataBuilder"]:
        return BitDecodeAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Return shape for regular KV cache (we'll pack it on-the-fly)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 2, 3, 4)


@dataclass
class BitDecodeAttentionMetadata:
    """Metadata for BitDecode attention."""
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    
    # BitDecode specific parameters
    quant_mode: str = "k-channel"  # or "k-token" 
    group_size: int = 128
    num_bits: int = 4
    pack_nums: int = 8  # 16 / num_bits


class BitDecodeAttentionMetadataBuilder(
    AttentionMetadataBuilder[BitDecodeAttentionMetadata]
):
    cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
        quant_mode: str = "k-channel",
        group_size: int = 128,
        num_bits: int = 4,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.quant_mode = quant_mode
        self.group_size = group_size
        self.num_bits = num_bits
        self.pack_nums = 16 // num_bits

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> BitDecodeAttentionMetadata:
        """
        Build BitDecode attention metadata from common metadata.
        
        Args:
            common_prefix_len: Length of common prefix for cascade attention (usually 0)
            common_attn_metadata: Common attention metadata shared across backends
            fast_build: Whether to use fast build (for speculative decoding)
        """
        # Simply return the common metadata wrapped in our metadata class
        # Since BitDecodeAttentionMetadata extends CommonAttentionMetadata,
        # we can just return the common_attn_metadata directly
        return common_attn_metadata


class BitDecodeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        quant_mode: str = "k-token",
        group_size: int = 128,
        num_bits: int = 4,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.quant_mode = quant_mode
        self.group_size = group_size
        self.num_bits = num_bits
        self.pack_nums = 16 // num_bits
        self.kv_cache_dtype = kv_cache_dtype
        
        # Pre-allocate quantized cache buffers (will be sized on first use)
        self.k_pack = None
        self.k_params = None
        self.v_pack = None
        self.v_params = None
        self.cached_batch_size = 0
        self.cached_seqlen_kv = 0
        
        logger.info(
            f"BitDecodeAttentionImpl initialized with quant_mode={quant_mode}, "
            f"group_size={group_size}, num_bits={num_bits}"
        )
    
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BitDecodeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with BitDecode quantized attention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run
            return output

        num_actual_tokens = attn_metadata.num_actual_tokens
        
        # Extract key and value cache
        key_cache, value_cache = kv_cache.unbind(0)
        
        # For benchmarking: assume we have the full KV cache ready
        # In practice, you'd need to handle incremental caching
        
        # Get batch size and sequence lengths from metadata
        batch_size = attn_metadata.seq_lens.shape[0]
        
        # Reshape KV cache for your kernel
        # This is simplified - you'll need proper block-based access
        seqlen_kv = num_actual_tokens
        
        # Initialize packed KV cache tensors
        device = query.device
        d = self.head_size
        
        k_pack = torch.zeros(
            (batch_size, int(seqlen_kv // self.pack_nums), self.num_kv_heads, d),
            dtype=torch.uint16,
            device=device
        )
        k_params = torch.zeros(
            (batch_size, int(seqlen_kv // self.group_size), self.num_kv_heads, d),
            dtype=torch.float32,
            device=device
        )
        
        v_pack = torch.zeros(
            (batch_size, seqlen_kv, self.num_kv_heads, int(d // self.pack_nums)),
            dtype=torch.uint16,
            device=device
        )
        v_params = torch.zeros(
            (batch_size, int(d // self.group_size), self.num_kv_heads, seqlen_kv),
            dtype=torch.float32,
            device=device
        )
        
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_kv, seqlen_kv, dtype=torch.int32, device=device
        )
        
        # Pack the KV cache
        # Note: You'll need to properly extract k_cache and v_cache from the paged cache
        # This is a simplified version assuming contiguous cache
        # k_cache_reshaped = key_cache[:, :seqlen_kv].reshape(batch_size, seqlen_kv, self.num_kv_heads, d)
        # v_cache_reshaped = value_cache[:, :seqlen_kv].reshape(batch_size, seqlen_kv, self.num_kv_heads, d)
        
        # kvcache_pack_int(
        #     k_cache_reshaped, k_pack, k_params,
        #     v_cache_reshaped, v_pack, v_params,
        #     None,  # opt_block_table
        #     cu_seqlens_k,
        #     seqlen_kv,
        #     self.quant_mode,
        #     self.group_size,
        #     self.num_bits
        # )
        
        # Reshape query for your kernel
        # query is [num_tokens, num_heads, head_size]
        # Reshape to [batch_size, seqlen_q, num_heads, head_size]
        seqlen_q = attn_metadata.max_query_len
        # q_reshaped = query[:num_actual_tokens].reshape(batch_size, seqlen_q, self.num_heads, d)
        
        q_pack = torch.zeros(
            (batch_size, 1, self.num_heads, d),
            dtype=torch.float16,
            device=device
        )

        # Run your quantized attention
        sm_scale = self.scale
        out_bitdecode = fwd_kvcache_int(
            q_pack,
            k_pack, k_params,
            v_pack, v_params,
            None,  # opt_block_table
            sm_scale,
            self.quant_mode,
            self.group_size,
            self.num_bits
        )
        
        # Reshape output back to [num_tokens, num_heads * head_size]
        # out_bitdecode is [batch_size, seqlen_q, num_heads, head_size]
        # We need to extract actual tokens, removing padding
        out_bitdecode = torch.zeros(
            (batch_size, seqlen_q, self.num_heads, d),
            dtype=torch.float16,
            device=device
        )
        token_idx = 0
        for i in range(batch_size):
            # Get actual query length for this sequence
            if i == 0:
                q_len = int(attn_metadata.query_start_loc_cpu[1].item())
            else:
                q_len = int(attn_metadata.query_start_loc_cpu[i+1].item() - 
                           attn_metadata.query_start_loc_cpu[i].item())
            
            # Copy actual tokens (without padding)
            # Keep shape as [q_len, num_heads, head_size]
            output[token_idx:token_idx + q_len] = out_bitdecode[i, :q_len]
            token_idx += q_len
        
        return output