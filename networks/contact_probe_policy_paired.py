from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.embeddings import SinusoidalPosEmb
from networks.contact_probe_policy_base import ProbeActionPolicyBase
from networks.embeddings import FlowMatchingTimestepPosEmb
from networks.residual_action_utils import (
    absolute_to_residual,
    build_residual_mask_from_absolute,
    build_residual_normalizer,
    build_residual_slot_indices,
    residual_to_absolute,
)
from networks.simple_diffusion import SimpleDDIMScheduler


# ------------------------------- Spatial attention -----------------------------------


class _SpatialAttention(nn.Module):
    # Option A positional encoding: query/key projections see the token feature concatenated
    # with its xyz (when that side is spatially grounded); value is feature-only. That keeps
    # relative geometry visible inside attention without polluting the propagated content.
    #
    # QKNorm (Henry et al., Chameleon, PaLM-variants): a per-head LayerNorm on Q and K right
    # before the dot product. Pins the scale of the attention logits regardless of how the
    # Q/K linear weights drift during long training, and cancels the scale mismatch between
    # LayerNorm-ed features (~unit variance) and raw world-xyz (~0.3 m). Cheap at this dim.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        query_has_xyz: bool,
        key_has_xyz: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.query_has_xyz = bool(query_has_xyz)
        self.key_has_xyz = bool(key_has_xyz)
        q_in = dim + (3 if self.query_has_xyz else 0)
        k_in = dim + (3 if self.key_has_xyz else 0)
        self.q_proj = nn.Linear(q_in, dim)
        self.k_proj = nn.Linear(k_in, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        # Per-head LayerNorm on Q and K: applied to the last dim (head_dim) so each head's
        # projected vector is renormalized before attention. Elementwise affine is off so
        # the norm keeps the logit scale at exactly head_dim**-0.5 — the softmax
        # temperature stays predictable throughout training.
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

    def forward(
        self,
        *,
        src: torch.Tensor,
        context: torch.Tensor,
        src_xyz: torch.Tensor | None,
        context_xyz: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, src_len, _ = src.shape
        ctx_len = context.shape[1]
        q_in = src if src_xyz is None else torch.cat([src, src_xyz.to(dtype=src.dtype)], dim=-1)
        k_in = context if context_xyz is None else torch.cat([context, context_xyz.to(dtype=context.dtype)], dim=-1)
        q = self.q_proj(q_in).reshape(batch_size, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k_in).reshape(batch_size, ctx_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(batch_size, ctx_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch_size, src_len, self.dim)
        return self.out_proj(out)


class _FFN(nn.Module):
    def __init__(self, *, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * float(mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialAttnBlock(nn.Module):
    # Pre-norm attention + residual + pre-norm FFN + residual. xyz stays outside the norm
    # so world coordinates are preserved exactly across layers.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        query_has_xyz: bool,
        key_has_xyz: bool,
        has_context: bool,
    ) -> None:
        super().__init__()
        self.has_context = bool(has_context)
        self.norm_src = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim) if self.has_context else None
        self.attn = _SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            query_has_xyz=query_has_xyz,
            key_has_xyz=key_has_xyz,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = _FFN(dim=dim, mlp_ratio=mlp_ratio)

    def forward(
        self,
        *,
        src: torch.Tensor,
        src_xyz: torch.Tensor | None,
        context: torch.Tensor | None = None,
        context_xyz: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.has_context:
            if context is None:
                raise ValueError("context required for cross-attention block")
            src_normed = self.norm_src(src)
            ctx_normed = self.norm_ctx(context)  # type: ignore[misc]
            attn_out = self.attn(
                src=src_normed,
                context=ctx_normed,
                src_xyz=src_xyz,
                context_xyz=context_xyz,
                attn_mask=attn_mask,
            )
        else:
            src_normed = self.norm_src(src)
            attn_out = self.attn(
                src=src_normed,
                context=src_normed,
                src_xyz=src_xyz,
                context_xyz=src_xyz,
                attn_mask=attn_mask,
            )
        x = src + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ------------------------------- Scene / action stages -------------------------------


class _PairedSceneEncoder(nn.Module):
    # Alternating block: self-attn(H), cross(C<-H), self-attn(C), cross(H<-C). All four
    # sublayers use Option A positional encoding: query/key carry the token's xyz.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.h_self_layers = nn.ModuleList()
        self.c_cross_layers = nn.ModuleList()
        self.c_self_layers = nn.ModuleList()
        self.h_cross_layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.h_self_layers.append(
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=True,
                    key_has_xyz=True,
                    has_context=False,
                )
            )
            self.c_cross_layers.append(
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=True,
                    key_has_xyz=True,
                    has_context=True,
                )
            )
            self.c_self_layers.append(
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=True,
                    key_has_xyz=True,
                    has_context=False,
                )
            )
            self.h_cross_layers.append(
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=True,
                    key_has_xyz=True,
                    has_context=True,
                )
            )

    def forward(
        self,
        *,
        hand_feat: torch.Tensor,
        hand_xyz: torch.Tensor,
        contact_feat: torch.Tensor,
        contact_xyz: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention on C needs the validity mask (broadcast on keys). H is dense (FPS'd)
        # so no mask. In cross-attention, the mask is on the keys: when C attends to H, no
        # mask; when H attends to C, we mask out invalid C keys.
        b = contact_feat.shape[0]
        n_c = contact_feat.shape[1]
        # (B, 1, n_c) broadcast over queries
        c_key_mask = contact_mask.to(dtype=torch.bool).view(b, 1, n_c)
        # Self-attn on C: keys=queries mask = contact_mask
        c_self_mask = c_key_mask
        for i in range(len(self.h_self_layers)):
            hand_feat = self.h_self_layers[i](src=hand_feat, src_xyz=hand_xyz, attn_mask=None)
            contact_feat = self.c_cross_layers[i](
                src=contact_feat,
                src_xyz=contact_xyz,
                context=hand_feat,
                context_xyz=hand_xyz,
                attn_mask=None,
            )
            contact_feat = self.c_self_layers[i](
                src=contact_feat,
                src_xyz=contact_xyz,
                attn_mask=c_self_mask,
            )
            hand_feat = self.h_cross_layers[i](
                src=hand_feat,
                src_xyz=hand_xyz,
                context=contact_feat,
                context_xyz=contact_xyz,
                attn_mask=c_key_mask,
            )
        return hand_feat, contact_feat


class _PairedActionDecoder(nn.Module):
    # Three learned action queries self-attend, then cross-attend into H (or H∪C). Each
    # _SpatialAttnBlock already wraps attention+FFN with pre-norm, so per-layer we get:
    # [SelfAttn+FFN]  ->  [CrossAttn+FFN]. That maps onto the spec's per-layer {SelfAttn,
    # CrossAttn, FFN} without stacking a third redundant FFN on top.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.self_layers = nn.ModuleList(
            [
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=False,
                    key_has_xyz=False,
                    has_context=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.cross_layers = nn.ModuleList(
            [
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=False,
                    key_has_xyz=True,
                    has_context=True,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        *,
        action_tokens: torch.Tensor,
        memory_feat: torch.Tensor,
        memory_xyz: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_mask = None
        if memory_mask is not None:
            attn_mask = memory_mask.to(dtype=torch.bool).unsqueeze(1)  # (B, 1, M)
        for self_layer, cross_layer in zip(self.self_layers, self.cross_layers):
            action_tokens = self_layer(src=action_tokens, src_xyz=None, attn_mask=None)
            action_tokens = cross_layer(
                src=action_tokens,
                src_xyz=None,
                context=memory_feat,
                context_xyz=memory_xyz,
                attn_mask=attn_mask,
            )
        return action_tokens


# ------------------------------- Contact-latent bottleneck ---------------------------


class _ContactOnlyEncoder(nn.Module):
    # Compress the raw contact cloud into a single CLS latent Z ∈ R^(B, D). The
    # contact tokens self-attend with xyz-aware attention (Option A), then a learned
    # CLS token (no xyz) cross-attends over the processed contacts. Permutation-
    # invariant over contacts by construction; padded contacts are masked out of both
    # the self-attention and the CLS cross-attention.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_proj = nn.Linear(6, int(dim))
        self.self_layers = nn.ModuleList(
            [
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=True,
                    key_has_xyz=True,
                    has_context=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, int(dim)) * 0.02)
        self.cls_attn = _SpatialAttnBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            query_has_xyz=False,
            key_has_xyz=True,
            has_context=True,
        )

    def forward(
        self,
        *,
        contact_feat_in: torch.Tensor,
        contact_xyz: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        # contact_feat_in: (B, N_c, 6). contact_xyz: (B, N_c, 3). contact_mask: (B, N_c) bool.
        # Returns Z ∈ R^(B, D).
        batch_size = contact_feat_in.shape[0]
        n_c = contact_feat_in.shape[1]
        contact_feat = self.input_proj(contact_feat_in)
        # Zero out padded contact tokens so they cannot leak via LayerNorm / FFN into the
        # valid-contact features before the attention mask filters them.
        contact_feat = contact_feat * contact_mask.unsqueeze(-1).to(dtype=contact_feat.dtype)
        key_mask = contact_mask.to(dtype=torch.bool).view(batch_size, 1, n_c)
        for layer in self.self_layers:
            contact_feat = layer(src=contact_feat, src_xyz=contact_xyz, attn_mask=key_mask)
        cls = self.cls_token.expand(batch_size, -1, -1).contiguous()
        cls = self.cls_attn(
            src=cls,
            src_xyz=None,
            context=contact_feat,
            context_xyz=contact_xyz,
            attn_mask=key_mask,
        )
        return cls.squeeze(1)


class _ContactSlotEncoder(nn.Module):
    # Compress the raw unordered contact cloud into M slot latents. Contacts still
    # self-attend in d_model space, but each learned slot cross-attends over all valid
    # contacts and is then projected down to slot_dim. Flattening M * slot_dim preserves
    # the public bottleneck size while making the bottleneck internally multi-token.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        num_slots: int,
        slot_dim: int,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        if self.slot_dim < 1:
            raise ValueError(f"slot_dim must be >= 1, got {slot_dim}")
        self.input_proj = nn.Linear(6, self.dim)
        self.self_layers = nn.ModuleList(
            [
                _SpatialAttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    query_has_xyz=True,
                    key_has_xyz=True,
                    has_context=False,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.dim) * 0.02)
        self.slot_cross = _SpatialAttnBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            query_has_xyz=False,
            key_has_xyz=True,
            has_context=True,
        )
        self.slot_self = _SpatialAttnBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            query_has_xyz=False,
            key_has_xyz=False,
            has_context=False,
        )
        self.slot_proj = nn.Linear(self.dim, self.slot_dim)

    def forward(
        self,
        *,
        contact_feat_in: torch.Tensor,
        contact_xyz: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Returns low-dimensional slots with shape (B, M, slot_dim).
        batch_size = contact_feat_in.shape[0]
        n_c = contact_feat_in.shape[1]
        contact_feat = self.input_proj(contact_feat_in)
        contact_feat = contact_feat * contact_mask.unsqueeze(-1).to(dtype=contact_feat.dtype)
        key_mask = contact_mask.to(dtype=torch.bool).view(batch_size, 1, n_c)
        for layer in self.self_layers:
            contact_feat = layer(src=contact_feat, src_xyz=contact_xyz, attn_mask=key_mask)
        slots = self.slot_queries.expand(batch_size, -1, -1).contiguous()
        slots = self.slot_cross(
            src=slots,
            src_xyz=None,
            context=contact_feat,
            context_xyz=contact_xyz,
            attn_mask=key_mask,
        )
        slots = self.slot_self(src=slots, src_xyz=None, attn_mask=None)
        return self.slot_proj(slots)


class _ClsUnpacker(nn.Module):
    # Project the CLS vector Z ∈ R^(B, D) into K' pseudo-tokens ∈ R^(B, K', D). K'==1
    # still applies a learned linear projection so the downstream attention sees a
    # transformed token rather than the raw CLS, and so the parameter count of the
    # "global context" path is comparable to the unpacked path's pre-interaction
    # processing.
    def __init__(self, *, dim: int, num_tokens: int, hidden_ratio: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        if self.num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if self.num_tokens == 1:
            self.proj = nn.Linear(self.dim, self.dim)
        else:
            hidden = max(self.dim, int(round(self.dim * float(hidden_ratio))))
            self.proj = nn.Sequential(
                nn.Linear(self.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.dim * self.num_tokens),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, D) → (B, K', D)
        if self.num_tokens == 1:
            return self.proj(z).unsqueeze(1)
        batch_size = z.shape[0]
        flat = self.proj(z)
        return flat.view(batch_size, self.num_tokens, self.dim)


class _BottleneckSceneEncoder(nn.Module):
    # Scene encoder that runs when the contact bottleneck is active. Two topologies:
    #
    #   K'==1: Z is concatenated into H's self-attention sequence every layer, so each
    #          H token's softmax ranges over both its H peers and the single global Z
    #          token — adaptive mixing at the attention layer itself, matching the
    #          "global context token injected into H's self-attention" recipe. Z gets
    #          a zero xyz pad inside the scene encoder so the Option-A xyz-aware
    #          attention can run over a uniform xyz shape; the zero is a static
    #          sentinel (no dependency on contact xyz), so the inference-time strict
    #          bottleneck is preserved.
    #
    #   K'>1:  Bidirectional paired H↔Z with Z-side xyz absent. Per layer
    #          [H-self → Z←H(q no-xyz, k xyz) → Z-self(no xyz) → H←Z(q xyz, k no-xyz)]
    #          — same four-sublayer structure as _PairedSceneEncoder.
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        num_z_tokens: int,
    ) -> None:
        super().__init__()
        self.num_z_tokens = int(num_z_tokens)
        if self.num_z_tokens < 1:
            raise ValueError(f"num_z_tokens must be >= 1, got {num_z_tokens}")
        if self.num_z_tokens == 1:
            # Single self-attn primitive over [H, Z] per layer — no separate H↔Z
            # cross paths. Z's xyz is a learned static pad (zero here); the feature
            # vector Z ∈ R^D is what actually carries the contact information.
            self.aug_self_layers = nn.ModuleList(
                [
                    _SpatialAttnBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        query_has_xyz=True,
                        key_has_xyz=True,
                        has_context=False,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.h_self_layers = None
            self.h_cross_layers = None
            self.z_cross_layers = None
            self.z_self_layers = None
        else:
            self.aug_self_layers = None
            self.h_self_layers = nn.ModuleList()
            self.h_cross_layers = nn.ModuleList()
            self.z_cross_layers = nn.ModuleList()
            self.z_self_layers = nn.ModuleList()
            for _ in range(int(num_layers)):
                self.h_self_layers.append(
                    _SpatialAttnBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        query_has_xyz=True,
                        key_has_xyz=True,
                        has_context=False,
                    )
                )
                self.h_cross_layers.append(
                    _SpatialAttnBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        query_has_xyz=True,
                        key_has_xyz=False,
                        has_context=True,
                    )
                )
                self.z_cross_layers.append(
                    _SpatialAttnBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        query_has_xyz=False,
                        key_has_xyz=True,
                        has_context=True,
                    )
                )
                self.z_self_layers.append(
                    _SpatialAttnBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        query_has_xyz=False,
                        key_has_xyz=False,
                        has_context=False,
                    )
                )

    def forward(
        self,
        *,
        hand_feat: torch.Tensor,
        hand_xyz: torch.Tensor,
        z_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_z_tokens == 1:
            assert self.aug_self_layers is not None
            batch_size = hand_feat.shape[0]
            n_h = hand_feat.shape[1]
            z_xyz_pad = torch.zeros(
                (batch_size, 1, 3), device=hand_feat.device, dtype=hand_xyz.dtype,
            )
            for layer in self.aug_self_layers:
                aug_feat = torch.cat([hand_feat, z_tokens], dim=1)
                aug_xyz = torch.cat([hand_xyz, z_xyz_pad], dim=1)
                aug_feat = layer(src=aug_feat, src_xyz=aug_xyz, attn_mask=None)
                hand_feat = aug_feat[:, :n_h]
                z_tokens = aug_feat[:, n_h:]
            return hand_feat, z_tokens
        assert self.h_self_layers is not None
        assert self.h_cross_layers is not None
        assert self.z_cross_layers is not None
        assert self.z_self_layers is not None
        for i in range(len(self.h_self_layers)):
            hand_feat = self.h_self_layers[i](src=hand_feat, src_xyz=hand_xyz, attn_mask=None)
            z_tokens = self.z_cross_layers[i](
                src=z_tokens,
                src_xyz=None,
                context=hand_feat,
                context_xyz=hand_xyz,
                attn_mask=None,
            )
            z_tokens = self.z_self_layers[i](src=z_tokens, src_xyz=None, attn_mask=None)
            hand_feat = self.h_cross_layers[i](
                src=hand_feat,
                src_xyz=hand_xyz,
                context=z_tokens,
                context_xyz=None,
                attn_mask=None,
            )
        return hand_feat, z_tokens


class _NonContactContextEncoder(nn.Module):
    """Encode no-touch points as coarse global avoidance tokens.

    Unlike contact tokens, these points describe where the whole manipulator should
    avoid collision. The encoder therefore produces a few summary tokens instead of
    preserving pointwise geometry for the scene encoder.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_tokens: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_tokens = int(num_tokens)
        if self.num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        self.input_proj = nn.Sequential(
            nn.Linear(3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.summary_queries = nn.Parameter(torch.randn(1, self.num_tokens, self.dim) * 0.02)
        self.summary_cross = _SpatialAttnBlock(
            dim=self.dim,
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            query_has_xyz=False,
            key_has_xyz=True,
            has_context=True,
        )

    def forward(
        self,
        *,
        non_contact_points: torch.Tensor,
        non_contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(non_contact_points.shape[0])
        n_points = int(non_contact_points.shape[1])
        if n_points == 0:
            return non_contact_points.new_zeros((batch_size, self.num_tokens, self.dim))

        valid = non_contact_mask.to(dtype=torch.bool)
        if valid.shape != (batch_size, n_points):
            raise ValueError(
                f"non_contact_mask must have shape ({batch_size}, {n_points}), got {tuple(valid.shape)}"
            )
        empty_rows = ~valid.any(dim=1)
        points = non_contact_points[..., :3].to(dtype=torch.float32)
        feat = self.input_proj(points)
        feat = feat * valid.unsqueeze(-1).to(dtype=feat.dtype)

        # scaled_dot_product_attention has undefined behavior for rows with no valid keys.
        # Use one zero feature as a harmless key, then zero the output for those rows.
        if bool(empty_rows.any()):
            valid = valid.clone()
            feat = feat.clone()
            valid[empty_rows, 0] = True
            feat[empty_rows, 0] = 0.0

        queries = self.summary_queries.expand(batch_size, -1, -1).contiguous()
        tokens = self.summary_cross(
            src=queries,
            src_xyz=None,
            context=feat,
            context_xyz=points,
            attn_mask=valid.view(batch_size, 1, n_points),
        )
        if bool(empty_rows.any()):
            tokens = tokens.masked_fill(empty_rows.view(batch_size, 1, 1), 0.0)
        return tokens


# ------------------------------- Paired policy ---------------------------------------


class ContactProbePolicyPairedTransformer(ProbeActionPolicyBase):
    _VALID_VARIANTS = {"flow_matching", "diffusion", "mlp"}

    def __init__(
        self,
        *,
        action_dim: int,
        variant: str,
        base_translation_indices: Sequence[int],
        base_rotation_indices: Sequence[int],
        pose_groups: Sequence[Dict[str, Sequence[int]]] | None = None,
        robot_names: tuple[str, ...] | list[str] | None = None,
        action_normalizer_scale: torch.Tensor | None = None,
        action_normalizer_offset: torch.Tensor | None = None,
        residual_normalizer_scale: torch.Tensor | None = None,
        residual_normalizer_offset: torch.Tensor | None = None,
        d_model: int = 256,
        num_heads: int = 8,
        num_scene_layers: int = 3,
        num_action_layers: int = 2,
        mlp_ratio: float = 4.0,
        hand_points: int = 32,
        use_contact_tokens_in_decoder: bool = False,
        step_embed_dim: int = 256,
        num_inference_steps: int = 10,
        flow_time_beta_alpha: float = 1.5,
        flow_time_beta_beta: float = 1.0,
        noise_scheduler: SimpleDDIMScheduler | None = None,
        contact_bottleneck_enabled: bool = False,
        contact_bottleneck_num_encoder_layers: int = 2,
        contact_bottleneck_unpack_tokens: int = 1,
        contact_bottleneck_unpack_mlp_ratio: float = 2.0,
        contact_bottleneck_mode: str = "cls_unpack",
        contact_bottleneck_slot_count: int | None = None,
        contact_bottleneck_slot_dim: int | None = None,
        action_tokenization: str = "grouped",
        use_qref_action_tokens: bool = False,
        use_qref_memory_tokens: bool = False,
        action_token_layer_norm: bool = False,
        non_contact_context_enabled: bool = False,
        non_contact_context_tokens: int = 4,
        non_contact_context_mlp_ratio: float = 1.0,
    ) -> None:
        super().__init__(
            action_dim=int(action_dim),
            robot_names=robot_names,
            action_normalizer_scale=action_normalizer_scale,
            action_normalizer_offset=action_normalizer_offset,
        )
        mode_key = str(variant).strip().lower()
        if mode_key.startswith("paired_transformer_"):
            mode_key = mode_key[len("paired_transformer_") :]
        if mode_key == "paired_transformer":
            # The plain alias defaults to diffusion. Select flow_matching / mlp explicitly
            # via paired_transformer_flow_matching / paired_transformer_mlp.
            mode_key = "diffusion"
        if mode_key not in self._VALID_VARIANTS:
            raise ValueError(f"Unsupported paired_transformer variant={variant}")
        self.variant = mode_key
        self.num_inference_steps = int(num_inference_steps)
        self.flow_time_beta_alpha = float(flow_time_beta_alpha)
        self.flow_time_beta_beta = float(flow_time_beta_beta)
        self.hand_points = int(hand_points)
        self.use_contact_tokens_in_decoder = bool(use_contact_tokens_in_decoder)
        self.d_model = int(d_model)
        action_tokenization_key = str(action_tokenization).strip().lower()
        if action_tokenization_key not in {"grouped", "per_joint"}:
            raise ValueError(
                f"Unsupported action_tokenization={action_tokenization!r}. "
                "Expected 'grouped' or 'per_joint'."
            )
        self.action_tokenization = action_tokenization_key
        self.use_qref_action_tokens = bool(use_qref_action_tokens)
        self.use_qref_memory_tokens = bool(use_qref_memory_tokens)
        self.action_token_layer_norm = bool(action_token_layer_norm)
        self.non_contact_context_enabled = bool(non_contact_context_enabled)
        self.non_contact_context_tokens = (
            int(non_contact_context_tokens) if self.non_contact_context_enabled else 0
        )

        self._slot = build_residual_slot_indices(
            dof=int(action_dim),
            base_translation_indices=base_translation_indices,
            base_rotation_indices=base_rotation_indices,
            pose_groups=pose_groups,
        )
        self.residual_dim = int(self._slot["residual_dim"])
        self.pose_groups = list(self._slot["pose_groups"])
        self.pose_residual_slices = list(self._slot["pose_residual_slices"])
        self.num_pose_groups = len(self.pose_groups)
        self.pose_token_count = 2 * self.num_pose_groups
        self.joint_token_offset = self.pose_token_count
        self.num_joint_slots = int(self._slot["joints_slice"][1]) - int(self._slot["joints_slice"][0])
        self._single_pose_action_io = self.num_pose_groups == 1

        # Residual normalizer lives alongside the base (absolute) normalizer so residual-space
        # losses and inference reuse the same affine plumbing without relying on the base class.
        num_robots = int(self.action_normalizer_scale.shape[0])
        if residual_normalizer_scale is None or residual_normalizer_offset is None:
            residual_normalizer_scale, residual_normalizer_offset = build_residual_normalizer(
                absolute_scale=self.action_normalizer_scale,
                absolute_offset=self.action_normalizer_offset,
                slot=self._slot,
            )
        residual_normalizer_scale = torch.as_tensor(residual_normalizer_scale, dtype=torch.float32)
        residual_normalizer_offset = torch.as_tensor(residual_normalizer_offset, dtype=torch.float32)
        if residual_normalizer_scale.shape != (num_robots, self.residual_dim):
            raise ValueError(
                f"residual_normalizer_scale must have shape ({num_robots}, {self.residual_dim}), "
                f"got {tuple(residual_normalizer_scale.shape)}"
            )
        if residual_normalizer_offset.shape != residual_normalizer_scale.shape:
            raise ValueError("residual normalizer scale/offset shape mismatch")
        self.register_buffer("residual_normalizer_scale", residual_normalizer_scale)
        self.register_buffer("residual_normalizer_offset", residual_normalizer_offset)

        # Phase 1: token projections. H projects (xyz, normal); C projects (xyz, normal). No
        # link-ID embeddings — pure geometry for both streams.
        self.hand_proj = nn.Linear(6, self.d_model)

        self.contact_bottleneck_enabled = bool(contact_bottleneck_enabled)
        self.contact_bottleneck_mode = str(contact_bottleneck_mode).strip().lower()
        if self.contact_bottleneck_mode not in {"cls_unpack", "slot"}:
            raise ValueError(
                f"Unsupported contact_bottleneck_mode={contact_bottleneck_mode!r}. "
                "Expected 'cls_unpack' or 'slot'."
            )
        self.contact_bottleneck_unpack_tokens = (
            int(contact_bottleneck_unpack_tokens) if self.contact_bottleneck_enabled else 0
        )
        self.contact_bottleneck_slot_count = 0
        self.contact_bottleneck_slot_dim = 0
        if self.contact_bottleneck_enabled:
            # Bottleneck path: raw C tokens are consumed by the contact-only encoder, then
            # only the Z latent (or its unpacked tokens) participates in the scene stage.
            # Skip building the raw contact projection and paired scene encoder — they
            # would be unused parameters under this path.
            if int(contact_bottleneck_unpack_tokens) < 1:
                raise ValueError(
                    "contact_bottleneck_unpack_tokens must be >= 1, got "
                    f"{contact_bottleneck_unpack_tokens}"
                )
            self.contact_proj = None
            self.scene_encoder = None
            if self.contact_bottleneck_mode == "slot":
                self.contact_bottleneck_slot_count = int(
                    contact_bottleneck_slot_count or contact_bottleneck_unpack_tokens
                )
                self.contact_bottleneck_slot_dim = int(
                    contact_bottleneck_slot_dim
                    if contact_bottleneck_slot_dim is not None
                    else self.d_model // max(1, self.contact_bottleneck_slot_count)
                )
                if self.contact_bottleneck_slot_count < 1:
                    raise ValueError(
                        f"contact_bottleneck_slot_count must be >= 1, got {contact_bottleneck_slot_count}"
                    )
                if self.contact_bottleneck_slot_count * self.contact_bottleneck_slot_dim != self.d_model:
                    raise ValueError(
                        "slot bottleneck expects slot_count * slot_dim to equal d_model so "
                        "encode_contact_latent keeps the same flattened latent size. Got "
                        f"{self.contact_bottleneck_slot_count} * {self.contact_bottleneck_slot_dim} "
                        f"!= {self.d_model}."
                    )
                self.contact_bottleneck_unpack_tokens = self.contact_bottleneck_slot_count
                self.contact_only_encoder = None
                self.cls_unpacker = None
                self.contact_slot_encoder = _ContactSlotEncoder(
                    dim=self.d_model,
                    num_heads=int(num_heads),
                    num_layers=int(contact_bottleneck_num_encoder_layers),
                    mlp_ratio=float(mlp_ratio),
                    num_slots=self.contact_bottleneck_slot_count,
                    slot_dim=self.contact_bottleneck_slot_dim,
                )
                self.slot_to_scene_proj = nn.Linear(self.contact_bottleneck_slot_dim, self.d_model)
            else:
                self.contact_bottleneck_slot_count = int(contact_bottleneck_unpack_tokens)
                self.contact_bottleneck_slot_dim = self.d_model
                self.contact_only_encoder = _ContactOnlyEncoder(
                    dim=self.d_model,
                    num_heads=int(num_heads),
                    num_layers=int(contact_bottleneck_num_encoder_layers),
                    mlp_ratio=float(mlp_ratio),
                )
                self.cls_unpacker = _ClsUnpacker(
                    dim=self.d_model,
                    num_tokens=int(contact_bottleneck_unpack_tokens),
                    hidden_ratio=float(contact_bottleneck_unpack_mlp_ratio),
                )
                self.contact_slot_encoder = None
                self.slot_to_scene_proj = None
            self.bottleneck_scene_encoder = _BottleneckSceneEncoder(
                dim=self.d_model,
                num_heads=int(num_heads),
                num_layers=int(num_scene_layers),
                mlp_ratio=float(mlp_ratio),
                num_z_tokens=int(self.contact_bottleneck_unpack_tokens),
            )
        else:
            self.contact_proj = nn.Linear(6, self.d_model)
            self.scene_encoder = _PairedSceneEncoder(
                dim=self.d_model,
                num_heads=int(num_heads),
                num_layers=int(num_scene_layers),
                mlp_ratio=float(mlp_ratio),
            )
            self.contact_only_encoder = None
            self.cls_unpacker = None
            self.contact_slot_encoder = None
            self.slot_to_scene_proj = None
            self.bottleneck_scene_encoder = None

        self.non_contact_context_encoder = (
            _NonContactContextEncoder(
                dim=self.d_model,
                num_heads=int(num_heads),
                num_tokens=int(non_contact_context_tokens),
                mlp_ratio=float(non_contact_context_mlp_ratio),
            )
            if self.non_contact_context_enabled
            else None
        )

        # Phase 2: action decoder. Each 6D pose contributes two tokens (xyz / rot).
        # The legacy single-pose grouped mode still has three tokens
        # (xyz / rot / all joints), preserving existing behavior.
        if self.action_tokenization == "grouped":
            self.action_queries = nn.Parameter(torch.randn(self.pose_token_count + 1, self.d_model) * 0.02)
            self.joint_identity_embed = None
        else:
            self.action_queries = nn.Parameter(torch.randn(self.pose_token_count, self.d_model) * 0.02)
            self.joint_identity_embed = (
                nn.Parameter(torch.randn(self.num_joint_slots, self.d_model) * 0.02)
                if self.num_joint_slots > 0
                else None
            )

        self.action_token_norm = (
            nn.LayerNorm(self.d_model) if self.action_token_layer_norm else nn.Identity()
        )
        if self.use_qref_action_tokens:
            if self._single_pose_action_io:
                self.qref_xyz_proj = nn.Linear(3, self.d_model)
                self.qref_rot_proj = nn.Linear(3, self.d_model)
                self.qref_xyz_projs = None
                self.qref_rot_projs = None
            else:
                self.qref_xyz_proj = None
                self.qref_rot_proj = None
                self.qref_xyz_projs = nn.ModuleList([nn.Linear(3, self.d_model) for _ in self.pose_groups])
                self.qref_rot_projs = nn.ModuleList([nn.Linear(3, self.d_model) for _ in self.pose_groups])
            self.qref_joint_proj = (
                nn.Sequential(
                    nn.Linear(1, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                if self.num_joint_slots > 0
                else None
            )
        else:
            self.qref_xyz_proj = None
            self.qref_rot_proj = None
            self.qref_xyz_projs = None
            self.qref_rot_projs = None
            self.qref_joint_proj = None

        if self.use_qref_memory_tokens:
            if self._single_pose_action_io:
                self.qref_memory_base_tokens = nn.Parameter(torch.randn(2, self.d_model) * 0.02)
                self.qref_memory_pose_tokens = None
                self.qref_memory_xyz_proj = nn.Linear(3, self.d_model)
                self.qref_memory_rot_proj = nn.Linear(3, self.d_model)
                self.qref_memory_xyz_projs = None
                self.qref_memory_rot_projs = None
            else:
                self.qref_memory_base_tokens = None
                self.qref_memory_pose_tokens = nn.Parameter(torch.randn(self.pose_token_count, self.d_model) * 0.02)
                self.qref_memory_xyz_proj = None
                self.qref_memory_rot_proj = None
                self.qref_memory_xyz_projs = nn.ModuleList([nn.Linear(3, self.d_model) for _ in self.pose_groups])
                self.qref_memory_rot_projs = nn.ModuleList([nn.Linear(3, self.d_model) for _ in self.pose_groups])
            self.qref_memory_joint_identity = (
                nn.Parameter(torch.randn(self.num_joint_slots, self.d_model) * 0.02)
                if self.num_joint_slots > 0
                else None
            )
            self.qref_memory_joint_proj = (
                nn.Sequential(
                    nn.Linear(1, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                if self.num_joint_slots > 0
                else None
            )
            self.qref_memory_norm = nn.LayerNorm(self.d_model)
        else:
            self.qref_memory_base_tokens = None
            self.qref_memory_pose_tokens = None
            self.qref_memory_joint_identity = None
            self.qref_memory_xyz_proj = None
            self.qref_memory_rot_proj = None
            self.qref_memory_xyz_projs = None
            self.qref_memory_rot_projs = None
            self.qref_memory_joint_proj = None
            self.qref_memory_norm = None

        if self.variant == "mlp":
            # No timestep, no noisy action — action queries go straight through the decoder.
            self.time_mlp = None
            self.noisy_xyz_proj = None
            self.noisy_rot_proj = None
            self.noisy_xyz_projs = None
            self.noisy_rot_projs = None
            self.noisy_joint_proj: nn.Module | None = None
        else:
            time_embedding = (
                FlowMatchingTimestepPosEmb(step_embed_dim)
                if self.variant == "flow_matching"
                else SinusoidalPosEmb(step_embed_dim)
            )
            self.time_mlp = nn.Sequential(
                time_embedding,
                nn.Linear(step_embed_dim, step_embed_dim * 4),
                nn.Mish(),
                nn.Linear(step_embed_dim * 4, self.d_model),
            )
            if self._single_pose_action_io:
                self.noisy_xyz_proj = nn.Linear(3, self.d_model)
                self.noisy_rot_proj = nn.Linear(6, self.d_model)
                self.noisy_xyz_projs = None
                self.noisy_rot_projs = None
            else:
                self.noisy_xyz_proj = None
                self.noisy_rot_proj = None
                self.noisy_xyz_projs = nn.ModuleList([nn.Linear(3, self.d_model) for _ in self.pose_groups])
                self.noisy_rot_projs = nn.ModuleList([nn.Linear(6, self.d_model) for _ in self.pose_groups])
            if self.num_joint_slots > 0:
                if self.action_tokenization == "per_joint":
                    self.noisy_joint_proj = nn.Sequential(
                        nn.Linear(1, self.d_model),
                        nn.SiLU(),
                        nn.Linear(self.d_model, self.d_model),
                    )
                else:
                    self.noisy_joint_proj = nn.Linear(self.num_joint_slots, self.d_model)
            else:
                self.noisy_joint_proj = None

        self.action_decoder = _PairedActionDecoder(
            dim=self.d_model,
            num_heads=int(num_heads),
            num_layers=int(num_action_layers),
            mlp_ratio=float(mlp_ratio),
        )

        # Three small heads, one per output slot. Output dims match the residual layout.
        if self._single_pose_action_io:
            self.head_xyz = nn.Linear(self.d_model, 3)
            self.head_rot = nn.Linear(self.d_model, 6)
            self.head_xyzs = None
            self.head_rots = None
        else:
            self.head_xyz = None
            self.head_rot = None
            self.head_xyzs = nn.ModuleList([nn.Linear(self.d_model, 3) for _ in self.pose_groups])
            self.head_rots = nn.ModuleList([nn.Linear(self.d_model, 6) for _ in self.pose_groups])
        if self.num_joint_slots > 0:
            if self.action_tokenization == "per_joint":
                self.head_joints = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, 1),
                )
            else:
                self.head_joints = nn.Linear(self.d_model, self.num_joint_slots)
        else:
            self.head_joints = None

        if self.variant == "diffusion":
            self.noise_scheduler = noise_scheduler or SimpleDDIMScheduler(
                num_train_timesteps=64,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = None

    # --------------------------------- helpers ---------------------------------------

    def _normalize_residual(self, residual: torch.Tensor, *, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return residual * scale.to(dtype=residual.dtype) + offset.to(dtype=residual.dtype)

    def _unnormalize_residual(
        self,
        residual_norm: torch.Tensor,
        *,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        return (residual_norm - offset.to(dtype=residual_norm.dtype)) / scale.to(dtype=residual_norm.dtype)

    def _get_residual_normalizer_params(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_index = self._build_robot_index(batch, batch_size=batch_size)
        scale = self.residual_normalizer_scale.index_select(0, robot_index)
        offset = self.residual_normalizer_offset.index_select(0, robot_index)
        return scale, offset

    @staticmethod
    def _apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x * mask.to(dtype=x.dtype)

    def _build_residual_valid_mask(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        action_valid = self._build_action_valid_mask(batch, batch_size=batch_size)
        return build_residual_mask_from_absolute(
            action_valid.to(dtype=torch.float32),
            slot=self._slot,
        ).to(dtype=torch.bool)

    def _build_residual_joint_mask(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if "contact_joint_mask" not in batch:
            raise KeyError("paired_transformer contact_joints supervision requires contact_joint_mask in batch")
        joint_mask_abs = batch["contact_joint_mask"].to(self.device, dtype=torch.float32)
        if int(joint_mask_abs.shape[0]) != int(batch_size) or int(joint_mask_abs.shape[1]) != self.action_dim:
            raise ValueError(
                f"contact_joint_mask must have shape (B, {self.action_dim}), got {tuple(joint_mask_abs.shape)}"
            )
        return build_residual_mask_from_absolute(joint_mask_abs, slot=self._slot).to(dtype=torch.bool)

    def _contact_latent_to_z_tokens(self, latent_z: torch.Tensor) -> torch.Tensor:
        if self.contact_bottleneck_mode == "slot":
            if self.slot_to_scene_proj is None:
                raise RuntimeError("slot_to_scene_proj missing in slot bottleneck mode")
            if latent_z.ndim == 2:
                expected_dim = self.contact_bottleneck_slot_count * self.contact_bottleneck_slot_dim
                if int(latent_z.shape[-1]) != expected_dim:
                    raise ValueError(
                        f"slot latent_z must have trailing dim {expected_dim}, got {tuple(latent_z.shape)}"
                    )
                slots = latent_z.reshape(
                    int(latent_z.shape[0]),
                    self.contact_bottleneck_slot_count,
                    self.contact_bottleneck_slot_dim,
                )
            elif latent_z.ndim == 3:
                expected_shape = (self.contact_bottleneck_slot_count, self.contact_bottleneck_slot_dim)
                if tuple(int(x) for x in latent_z.shape[1:]) != expected_shape:
                    raise ValueError(
                        f"slot latent_z must have shape (B, {expected_shape[0]}, {expected_shape[1]}), "
                        f"got {tuple(latent_z.shape)}"
                    )
                slots = latent_z
            else:
                raise ValueError(f"slot latent_z must be rank 2 or 3, got {tuple(latent_z.shape)}")
            return self.slot_to_scene_proj(slots)

        if self.cls_unpacker is None:
            raise RuntimeError("cls_unpacker missing in cls_unpack bottleneck mode")
        if latent_z.ndim != 2 or int(latent_z.shape[-1]) != int(self.d_model):
            raise ValueError(
                f"latent_z must have shape (B, {self.d_model}) in cls_unpack mode, got {tuple(latent_z.shape)}"
            )
        return self.cls_unpacker(latent_z)

    def _build_qref_memory_tokens(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        if not self.use_qref_memory_tokens:
            return None
        if batch is None:
            raise ValueError("batch is required when use_qref_memory_tokens=true")
        q1_padded = batch["q1_padded"].to(self.device, dtype=torch.float32)
        if int(q1_padded.shape[0]) != int(batch_size):
            raise ValueError(
                f"q1_padded batch dim must be {batch_size}, got {tuple(q1_padded.shape)}"
            )
        action_scale, action_offset = self._get_action_normalizer_params(batch, batch_size=batch_size)
        qref_norm = self._normalize_action(q1_padded, scale=action_scale, offset=action_offset)
        action_valid = self._build_action_valid_mask(batch, batch_size=batch_size)
        qref_norm = self._apply_action_valid_mask(qref_norm, action_valid)
        joint_idx = list(self._slot["joint_indices"])

        if self._single_pose_action_io:
            if not isinstance(self.qref_memory_base_tokens, nn.Parameter):
                raise RuntimeError("qref_memory_base_tokens missing")
            if self.qref_memory_xyz_proj is None or self.qref_memory_rot_proj is None:
                raise RuntimeError("qref memory base projections missing")
            trans_idx = list(self.pose_groups[0]["translation_indices"])
            rot_idx = list(self.pose_groups[0]["rotation_indices"])
            pose_tokens = self.qref_memory_base_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()
            pose_tokens[:, 0] = pose_tokens[:, 0] + self.qref_memory_xyz_proj(qref_norm[:, trans_idx])
            pose_tokens[:, 1] = pose_tokens[:, 1] + self.qref_memory_rot_proj(qref_norm[:, rot_idx])
        else:
            if not isinstance(self.qref_memory_pose_tokens, nn.Parameter):
                raise RuntimeError("qref_memory_pose_tokens missing")
            if self.qref_memory_xyz_projs is None or self.qref_memory_rot_projs is None:
                raise RuntimeError("qref memory pose projections missing")
            pose_tokens = self.qref_memory_pose_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()
            for pose_idx, group in enumerate(self.pose_groups):
                trans_idx = list(group["translation_indices"])
                rot_idx = list(group["rotation_indices"])
                pose_tokens[:, 2 * pose_idx] = (
                    pose_tokens[:, 2 * pose_idx]
                    + self.qref_memory_xyz_projs[pose_idx](qref_norm[:, trans_idx])
                )
                pose_tokens[:, 2 * pose_idx + 1] = (
                    pose_tokens[:, 2 * pose_idx + 1]
                    + self.qref_memory_rot_projs[pose_idx](qref_norm[:, rot_idx])
                )
        tokens = [pose_tokens]
        if self.num_joint_slots > 0:
            if self.qref_memory_joint_identity is None or self.qref_memory_joint_proj is None:
                raise RuntimeError("qref memory joint modules missing")
            joint_tokens = self.qref_memory_joint_identity.unsqueeze(0).expand(batch_size, -1, -1)
            joint_tokens = joint_tokens + self.qref_memory_joint_proj(qref_norm[:, joint_idx].unsqueeze(-1))
            tokens.append(joint_tokens)
        qref_tokens = torch.cat(tokens, dim=1)
        if self.qref_memory_norm is not None:
            qref_tokens = self.qref_memory_norm(qref_tokens)
        return qref_tokens

    def _assemble_scene_memory(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        hand_feat: torch.Tensor,
        hand_xyz: torch.Tensor,
        contact_feat: torch.Tensor,
        contact_xyz: torch.Tensor | None,
        contact_mask: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(hand_feat.shape[0])
        n_h = int(hand_feat.shape[1])
        feat_parts = [hand_feat]
        xyz_parts = [hand_xyz]
        mask_parts: list[torch.Tensor] | None = None

        if self.use_contact_tokens_in_decoder:
            n_c = int(contact_feat.shape[1])
            if contact_xyz is None:
                contact_xyz_eff = torch.zeros(
                    (batch_size, n_c, 3), device=contact_feat.device, dtype=hand_xyz.dtype,
                )
            else:
                contact_xyz_eff = contact_xyz
            if contact_mask is None:
                contact_mask_eff = torch.ones(
                    (batch_size, n_c), dtype=torch.bool, device=contact_feat.device,
                )
            else:
                contact_mask_eff = contact_mask
            feat_parts.append(contact_feat)
            xyz_parts.append(contact_xyz_eff)
            hand_mask = torch.ones((batch_size, n_h), dtype=torch.bool, device=hand_feat.device)
            mask_parts = [hand_mask, contact_mask_eff]

        qref_tokens = self._build_qref_memory_tokens(batch, batch_size=batch_size)
        if qref_tokens is not None:
            n_q = int(qref_tokens.shape[1])
            qref_xyz = torch.zeros((batch_size, n_q, 3), device=hand_feat.device, dtype=hand_xyz.dtype)
            qref_mask = torch.ones((batch_size, n_q), dtype=torch.bool, device=hand_feat.device)
            feat_parts.append(qref_tokens)
            xyz_parts.append(qref_xyz)
            if mask_parts is None:
                hand_mask = torch.ones((batch_size, n_h), dtype=torch.bool, device=hand_feat.device)
                mask_parts = [hand_mask]
            mask_parts.append(qref_mask)

        if self.non_contact_context_encoder is not None:
            non_contact_cloud = batch.get("non_contact_cloud")
            non_contact_valid = batch.get("non_contact_valid_mask")
            if non_contact_cloud is None:
                non_contact_cloud = torch.zeros(
                    (batch_size, 0, 3), dtype=torch.float32, device=hand_feat.device,
                )
            else:
                non_contact_cloud = non_contact_cloud.to(hand_feat.device, dtype=torch.float32)
            if int(non_contact_cloud.shape[0]) != batch_size or int(non_contact_cloud.shape[-1]) < 3:
                raise ValueError(
                    "non_contact_cloud must have shape (B, N, >=3), got "
                    f"{tuple(non_contact_cloud.shape)} for batch_size={batch_size}"
                )
            n_nc = int(non_contact_cloud.shape[1])
            if non_contact_valid is None:
                non_contact_valid = torch.ones(
                    (batch_size, n_nc), dtype=torch.bool, device=hand_feat.device,
                )
            else:
                non_contact_valid = non_contact_valid.to(hand_feat.device).bool()
            if tuple(non_contact_valid.shape) != (batch_size, n_nc):
                raise ValueError(
                    f"non_contact_valid_mask must have shape ({batch_size}, {n_nc}), "
                    f"got {tuple(non_contact_valid.shape)}"
                )
            nc_tokens = self.non_contact_context_encoder(
                non_contact_points=non_contact_cloud[..., :3],
                non_contact_mask=non_contact_valid,
            )
            n_nc_tokens = int(nc_tokens.shape[1])
            nc_xyz = torch.zeros(
                (batch_size, n_nc_tokens, 3), device=hand_feat.device, dtype=hand_xyz.dtype,
            )
            nc_mask = non_contact_valid.any(dim=1, keepdim=True).expand(-1, n_nc_tokens)
            feat_parts.append(nc_tokens)
            xyz_parts.append(nc_xyz)
            if mask_parts is None:
                hand_mask = torch.ones((batch_size, n_h), dtype=torch.bool, device=hand_feat.device)
                mask_parts = [hand_mask]
            mask_parts.append(nc_mask)

        return {
            "memory_feat": torch.cat(feat_parts, dim=1),
            "memory_xyz": torch.cat(xyz_parts, dim=1),
            "memory_mask": torch.cat(mask_parts, dim=1) if mask_parts is not None else None,
        }

    # ------------------------------ scene encoding -----------------------------------

    def _encode_scene(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # Returns: hand_feat (B, N_h, D), hand_xyz (B, N_h, 3),
        #          contact_feat (B, N_c, D) — or Z_tokens (B, K', D) in bottleneck mode,
        #          contact_xyz (B, N_c, 3)   — or None in bottleneck mode (Z has no xyz),
        #          contact_mask (B, N_c)     — or None in bottleneck mode (all valid).
        # The decoder reuses this tuple across denoising steps.
        contact_cloud = batch["contact_cloud"].to(self.device, dtype=torch.float32)
        contact_valid = batch["contact_valid_mask"].to(self.device).bool()
        if "hand_points_world" not in batch or "hand_normals_world" not in batch:
            raise KeyError(
                "paired_transformer requires hand_points_world and hand_normals_world in the batch. "
                "The trainer must run FK on q1 and inject FPS-sampled hand surface points/normals."
            )
        hand_xyz = batch["hand_points_world"].to(self.device, dtype=torch.float32)
        hand_normals = batch["hand_normals_world"].to(self.device, dtype=torch.float32)
        if hand_xyz.shape[:2] != hand_normals.shape[:2]:
            raise ValueError("hand_points_world and hand_normals_world shapes disagree")

        contact_feat_in = contact_cloud[..., :6]
        contact_xyz = contact_cloud[..., :3]
        hand_feat_in = torch.cat([hand_xyz, hand_normals], dim=-1)

        hand_feat = self.hand_proj(hand_feat_in)

        if self.contact_bottleneck_enabled:
            assert self.bottleneck_scene_encoder is not None
            if self.contact_bottleneck_mode == "slot":
                assert self.contact_slot_encoder is not None
                assert self.slot_to_scene_proj is not None
                z_slots = self.contact_slot_encoder(
                    contact_feat_in=contact_feat_in,
                    contact_xyz=contact_xyz,
                    contact_mask=contact_valid,
                )
                z_tokens = self.slot_to_scene_proj(z_slots)
            else:
                assert self.contact_only_encoder is not None
                z = self.contact_only_encoder(
                    contact_feat_in=contact_feat_in,
                    contact_xyz=contact_xyz,
                    contact_mask=contact_valid,
                )
                z_tokens = self._contact_latent_to_z_tokens(z)
            hand_feat, z_tokens = self.bottleneck_scene_encoder(
                hand_feat=hand_feat,
                hand_xyz=hand_xyz,
                z_tokens=z_tokens,
            )
            return hand_feat, hand_xyz, z_tokens, None, None

        assert self.contact_proj is not None
        assert self.scene_encoder is not None
        contact_feat = self.contact_proj(contact_feat_in)
        # Zero out padded contact tokens so they cannot leak into the dense H stream before
        # the attention mask filters them.
        contact_feat = contact_feat * contact_valid.unsqueeze(-1).to(dtype=contact_feat.dtype)

        hand_feat, contact_feat = self.scene_encoder(
            hand_feat=hand_feat,
            hand_xyz=hand_xyz,
            contact_feat=contact_feat,
            contact_xyz=contact_xyz,
            contact_mask=contact_valid,
        )
        return hand_feat, hand_xyz, contact_feat, contact_xyz, contact_valid

    def encode_contact_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Run only the contact-only encoder to produce the strict contact bottleneck.
        # In cls_unpack mode this is Z ∈ R^(B, D). In slot mode the public API stays
        # flattened at R^(B, D), with D = M * slot_dim.
        if not self.contact_bottleneck_enabled:
            raise RuntimeError(
                "encode_contact_latent requires model.contact_bottleneck.enabled=true"
            )
        contact_cloud = batch["contact_cloud"].to(self.device, dtype=torch.float32)
        contact_valid = batch["contact_valid_mask"].to(self.device).bool()
        contact_feat_in = contact_cloud[..., :6]
        contact_xyz = contact_cloud[..., :3]
        if self.contact_bottleneck_mode == "slot":
            assert self.contact_slot_encoder is not None
            z_slots = self.contact_slot_encoder(
                contact_feat_in=contact_feat_in,
                contact_xyz=contact_xyz,
                contact_mask=contact_valid,
            )
            return z_slots.reshape(z_slots.shape[0], -1)
        assert self.contact_only_encoder is not None
        return self.contact_only_encoder(
            contact_feat_in=contact_feat_in,
            contact_xyz=contact_xyz,
            contact_mask=contact_valid,
        )

    def _build_scene_memory_from_latent(
        self,
        latent_z: torch.Tensor,
        *,
        batch: Dict[str, torch.Tensor],
        hand_xyz: torch.Tensor,
        hand_normals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Mirror of _build_scene_memory but the contact stream is replaced by a precomputed
        # bottleneck latent Z (e.g. predicted by the high-level diffusion policy). The hand
        # branch is unchanged: project (xyz, normal) to d_model, then run the bottleneck
        # scene encoder so H tokens get to attend over the unpacked Z tokens just as they
        # would during normal training.
        if not self.contact_bottleneck_enabled:
            raise RuntimeError(
                "_build_scene_memory_from_latent requires model.contact_bottleneck.enabled=true"
            )
        assert self.bottleneck_scene_encoder is not None
        if hand_xyz.shape[:-1] != hand_normals.shape[:-1]:
            raise ValueError(
                f"hand_xyz and hand_normals leading shape mismatch: {tuple(hand_xyz.shape)} vs {tuple(hand_normals.shape)}"
            )
        hand_feat_in = torch.cat([hand_xyz, hand_normals], dim=-1)
        hand_feat = self.hand_proj(hand_feat_in)
        z_tokens = self._contact_latent_to_z_tokens(latent_z)
        hand_feat, z_tokens = self.bottleneck_scene_encoder(
            hand_feat=hand_feat,
            hand_xyz=hand_xyz,
            z_tokens=z_tokens,
        )
        return self._assemble_scene_memory(
            batch=batch,
            hand_feat=hand_feat,
            hand_xyz=hand_xyz,
            contact_feat=z_tokens,
            contact_xyz=None,
            contact_mask=None,
        )

    @torch.no_grad()
    def predict_action_from_contact_latent(
        self,
        latent_z: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Decode-only counterpart to encode_contact_latent. The high-level pipeline predicts
        # Z ∈ R^(B, D) directly, so we skip the low-level's contact encoder and feed Z
        # straight into the unpacker → bottleneck scene encoder → action decoder pipeline.
        # batch must carry q1_padded, hand_points_world, hand_normals_world (the trainer
        # ordinarily synthesizes these), plus standard action-mask fields.
        if not self.contact_bottleneck_enabled:
            raise RuntimeError(
                "predict_action_from_contact_latent requires model.contact_bottleneck.enabled=true"
            )
        if "q1_padded" not in batch or "hand_points_world" not in batch or "hand_normals_world" not in batch:
            raise KeyError(
                "predict_action_from_contact_latent requires q1_padded, hand_points_world, "
                "and hand_normals_world in the batch."
            )
        q1_padded = batch["q1_padded"].to(self.device, dtype=torch.float32)
        hand_xyz = batch["hand_points_world"].to(self.device, dtype=torch.float32)
        hand_normals = batch["hand_normals_world"].to(self.device, dtype=torch.float32)
        latent_z = latent_z.to(self.device, dtype=torch.float32)
        if latent_z.ndim == 1:
            latent_z = latent_z.unsqueeze(0)
        batch_size = int(latent_z.shape[0])
        if int(q1_padded.shape[0]) != batch_size or int(hand_xyz.shape[0]) != batch_size:
            raise ValueError(
                f"latent_z, q1_padded, and hand_points_world must share batch dim, got "
                f"latent_z={tuple(latent_z.shape)} q1={tuple(q1_padded.shape)} hand={tuple(hand_xyz.shape)}"
            )
        action_valid_mask = self._build_action_valid_mask(batch, batch_size=batch_size)
        residual_valid_mask = self._build_residual_valid_mask(batch, batch_size=batch_size)
        res_scale, res_offset = self._get_residual_normalizer_params(batch, batch_size=batch_size)
        scene_memory = self._build_scene_memory_from_latent(
            latent_z, batch=batch, hand_xyz=hand_xyz, hand_normals=hand_normals,
        )

        if self.variant == "diffusion":
            if self.noise_scheduler is None:
                raise RuntimeError("Diffusion variant requires a noise scheduler")
            residual_norm = torch.randn((batch_size, self.residual_dim), device=self.device, dtype=torch.float32)
            residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for timestep in self.noise_scheduler.timesteps:
                model_output = self._predict_once(
                    batch=batch,
                    batch_size=batch_size,
                    residual_norm=residual_norm,
                    timestep=timestep,
                    scene_memory=scene_memory,
                )
                residual_norm = self.noise_scheduler.step(model_output, timestep, residual_norm).prev_sample
                residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
        elif self.variant == "flow_matching":
            residual_norm = torch.randn((batch_size, self.residual_dim), device=self.device, dtype=torch.float32)
            residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
            dt = 1.0 / float(self.num_inference_steps)
            for step_idx in range(self.num_inference_steps):
                t_value = 1.0 - float(step_idx) * dt
                timestep = torch.full((batch_size,), t_value, device=self.device, dtype=torch.float32)
                flow_pred = self._predict_once(
                    batch=batch,
                    batch_size=batch_size,
                    residual_norm=residual_norm,
                    timestep=timestep,
                    scene_memory=scene_memory,
                )
                residual_norm = residual_norm - dt * flow_pred
                residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
        else:
            residual_norm = self._predict_once(
                batch=batch,
                batch_size=batch_size,
                residual_norm=None,
                timestep=None,
                scene_memory=scene_memory,
            )
            residual_norm = self._apply_mask(residual_norm, residual_valid_mask)

        residual = self._unnormalize_residual(residual_norm, scale=res_scale, offset=res_offset)
        residual = self._apply_mask(residual, residual_valid_mask)
        action_abs = residual_to_absolute(residual, q1_padded, slot=self._slot)
        action_abs = self._apply_action_valid_mask(action_abs, action_valid_mask)
        return {
            "action": action_abs,
            "action_pred": action_abs,
            "residual_pred": residual,
            "residual_norm_pred": residual_norm,
        }

    # ------------------------------ action decoding ----------------------------------

    def _build_action_input(
        self,
        *,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        residual_norm: torch.Tensor | None,
        timestep: torch.Tensor | float | int | None,
    ) -> torch.Tensor:
        if self.action_tokenization == "per_joint":
            base_tokens = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
            if self.num_joint_slots > 0:
                if self.joint_identity_embed is None:
                    raise RuntimeError("joint_identity_embed missing in per_joint mode")
                joint_tokens = self.joint_identity_embed.unsqueeze(0).expand(batch_size, -1, -1)
                action_tokens = torch.cat([base_tokens, joint_tokens], dim=1).contiguous()
            else:
                action_tokens = base_tokens.contiguous()
        else:
            action_tokens = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        if self.use_qref_action_tokens:
            if batch is None:
                raise ValueError("batch is required when use_qref_action_tokens=true")
            q1_padded = batch["q1_padded"].to(self.device, dtype=torch.float32)
            if int(q1_padded.shape[0]) != int(batch_size):
                raise ValueError(
                    f"q1_padded batch dim must be {batch_size}, got {tuple(q1_padded.shape)}"
                )
            action_scale, action_offset = self._get_action_normalizer_params(batch, batch_size=batch_size)
            qref_norm = self._normalize_action(q1_padded, scale=action_scale, offset=action_offset)
            action_valid = self._build_action_valid_mask(batch, batch_size=batch_size)
            qref_norm = self._apply_action_valid_mask(qref_norm, action_valid)
            joint_idx = list(self._slot["joint_indices"])
            action_tokens = action_tokens.clone()
            if self._single_pose_action_io:
                if self.qref_xyz_proj is not None:
                    trans_idx = list(self.pose_groups[0]["translation_indices"])
                    action_tokens[:, 0] = action_tokens[:, 0] + self.qref_xyz_proj(qref_norm[:, trans_idx])
                if self.qref_rot_proj is not None:
                    rot_idx = list(self.pose_groups[0]["rotation_indices"])
                    action_tokens[:, 1] = action_tokens[:, 1] + self.qref_rot_proj(qref_norm[:, rot_idx])
            else:
                if self.qref_xyz_projs is None or self.qref_rot_projs is None:
                    raise RuntimeError("qref pose projections missing")
                for pose_idx, group in enumerate(self.pose_groups):
                    trans_idx = list(group["translation_indices"])
                    rot_idx = list(group["rotation_indices"])
                    action_tokens[:, 2 * pose_idx] = (
                        action_tokens[:, 2 * pose_idx]
                        + self.qref_xyz_projs[pose_idx](qref_norm[:, trans_idx])
                    )
                    action_tokens[:, 2 * pose_idx + 1] = (
                        action_tokens[:, 2 * pose_idx + 1]
                        + self.qref_rot_projs[pose_idx](qref_norm[:, rot_idx])
                    )
            if self.qref_joint_proj is not None and self.num_joint_slots > 0:
                joint_qref = qref_norm[:, joint_idx].unsqueeze(-1)
                if self.action_tokenization == "per_joint":
                    action_tokens[:, self.joint_token_offset :] = (
                        action_tokens[:, self.joint_token_offset :] + self.qref_joint_proj(joint_qref)
                    )
                else:
                    action_tokens[:, self.joint_token_offset] = (
                        action_tokens[:, self.joint_token_offset] + self.qref_joint_proj(joint_qref).mean(dim=1)
                    )

        if self.variant == "mlp":
            return self.action_token_norm(action_tokens)
        if timestep is None:
            raise ValueError("timestep is required for non-mlp variants")
        if self.time_mlp is None:
            raise RuntimeError("time_mlp missing for non-mlp variant")
        t_tensor = timestep
        if not torch.is_tensor(t_tensor):
            t_tensor = torch.tensor([float(t_tensor)], device=self.device, dtype=torch.float32)
        elif t_tensor.ndim == 0:
            t_tensor = t_tensor[None]
        t_tensor = t_tensor.to(device=self.device, dtype=torch.float32).expand(batch_size)
        time_feat = self.time_mlp(t_tensor)  # (B, D)
        action_tokens = action_tokens + time_feat.unsqueeze(1)

        if residual_norm is not None:
            if residual_norm.shape != (batch_size, self.residual_dim):
                raise ValueError(
                    f"residual_norm must have shape ({batch_size}, {self.residual_dim}), "
                    f"got {tuple(residual_norm.shape)}"
                )
            action_tokens = action_tokens.clone()
            if self._single_pose_action_io:
                xyz_emb = self.noisy_xyz_proj(residual_norm[:, 0:3])  # type: ignore[misc]
                rot_emb = self.noisy_rot_proj(residual_norm[:, 3:9])  # type: ignore[misc]
                action_tokens[:, 0] = action_tokens[:, 0] + xyz_emb
                action_tokens[:, 1] = action_tokens[:, 1] + rot_emb
            else:
                if self.noisy_xyz_projs is None or self.noisy_rot_projs is None:
                    raise RuntimeError("noisy pose projections missing")
                for pose_idx, slices in enumerate(self.pose_residual_slices):
                    xyz_start, xyz_end = slices["xyz_slice"]
                    rot_start, rot_end = slices["rot6d_slice"]
                    xyz_emb = self.noisy_xyz_projs[pose_idx](residual_norm[:, int(xyz_start) : int(xyz_end)])
                    rot_emb = self.noisy_rot_projs[pose_idx](residual_norm[:, int(rot_start) : int(rot_end)])
                    action_tokens[:, 2 * pose_idx] = action_tokens[:, 2 * pose_idx] + xyz_emb
                    action_tokens[:, 2 * pose_idx + 1] = action_tokens[:, 2 * pose_idx + 1] + rot_emb
            if self.noisy_joint_proj is not None and self.num_joint_slots > 0:
                joint_start, joint_end = self._slot["joints_slice"]
                if self.action_tokenization == "per_joint":
                    joint_emb = self.noisy_joint_proj(
                        residual_norm[:, int(joint_start) : int(joint_end)].unsqueeze(-1)
                    )
                    action_tokens[:, self.joint_token_offset :] = (
                        action_tokens[:, self.joint_token_offset :] + joint_emb
                    )
                else:
                    joint_emb = self.noisy_joint_proj(residual_norm[:, int(joint_start) : int(joint_end)])
                    action_tokens[:, self.joint_token_offset] = (
                        action_tokens[:, self.joint_token_offset] + joint_emb
                    )
        return self.action_token_norm(action_tokens)

    def _decode_residual(
        self,
        *,
        action_tokens: torch.Tensor,
        scene_memory: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        memory_feat = scene_memory["memory_feat"]
        memory_xyz = scene_memory["memory_xyz"]
        memory_mask = scene_memory.get("memory_mask")
        out = self.action_decoder(
            action_tokens=action_tokens,
            memory_feat=memory_feat,
            memory_xyz=memory_xyz,
            memory_mask=memory_mask,
        )
        pose_chunks: list[torch.Tensor] = []
        if self._single_pose_action_io:
            if self.head_xyz is None or self.head_rot is None:
                raise RuntimeError("single-pose residual heads missing")
            pose_chunks.extend([self.head_xyz(out[:, 0]), self.head_rot(out[:, 1])])
        else:
            if self.head_xyzs is None or self.head_rots is None:
                raise RuntimeError("multi-pose residual heads missing")
            for pose_idx in range(self.num_pose_groups):
                pose_chunks.append(self.head_xyzs[pose_idx](out[:, 2 * pose_idx]))
                pose_chunks.append(self.head_rots[pose_idx](out[:, 2 * pose_idx + 1]))
        if self.head_joints is not None and self.num_joint_slots > 0:
            if self.action_tokenization == "per_joint":
                delta_joint = self.head_joints(out[:, self.joint_token_offset :]).squeeze(-1)
            else:
                delta_joint = self.head_joints(out[:, self.joint_token_offset])
        else:
            delta_joint = out.new_zeros((out.shape[0], 0))
        pose_chunks.append(delta_joint)
        return torch.cat(pose_chunks, dim=-1)

    def _build_scene_memory(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        hand_feat, hand_xyz, contact_feat, contact_xyz, contact_mask = self._encode_scene(batch)
        return self._assemble_scene_memory(
            batch=batch,
            hand_feat=hand_feat,
            hand_xyz=hand_xyz,
            contact_feat=contact_feat,
            contact_xyz=contact_xyz,
            contact_mask=contact_mask,
        )

    def _predict_once(
        self,
        *,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        residual_norm: torch.Tensor | None,
        timestep: torch.Tensor | float | int | None,
        scene_memory: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action_tokens = self._build_action_input(
            batch=batch,
            batch_size=batch_size,
            residual_norm=residual_norm,
            timestep=timestep,
        )
        return self._decode_residual(action_tokens=action_tokens, scene_memory=scene_memory)

    # ------------------------------ flow-matching sampler ----------------------------

    def _sample_flow_matching_timesteps(self, batch_size: int, *, dtype: torch.dtype) -> torch.Tensor:
        gamma1 = torch.empty((batch_size,), dtype=dtype, device=self.device).uniform_(0, 1).pow(
            1.0 / float(self.flow_time_beta_alpha)
        )
        gamma2 = torch.empty((batch_size,), dtype=dtype, device=self.device).uniform_(0, 1).pow(
            1.0 / float(self.flow_time_beta_beta)
        )
        time_beta = gamma1 / (gamma1 + gamma2)
        return time_beta * 0.999 + 0.001

    # ---------------------------------- loss -----------------------------------------

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]] | tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        action_abs = batch["action"].to(self.device, dtype=torch.float32)
        q1_padded = batch["q1_padded"].to(self.device, dtype=torch.float32)
        batch_size = int(action_abs.shape[0])
        action_valid_mask = self._build_action_valid_mask(batch, batch_size=batch_size)
        residual_valid_mask = self._build_residual_valid_mask(batch, batch_size=batch_size)

        res_scale, res_offset = self._get_residual_normalizer_params(batch, batch_size=batch_size)
        residual_target = absolute_to_residual(action_abs, q1_padded, slot=self._slot)
        residual_target = self._apply_mask(residual_target, residual_valid_mask)
        residual_target_norm = self._normalize_residual(residual_target, scale=res_scale, offset=res_offset)
        residual_target_norm = self._apply_mask(residual_target_norm, residual_valid_mask)

        scene_memory = self._build_scene_memory(batch)
        pred_residual_norm: torch.Tensor

        if self.variant == "diffusion":
            if self.noise_scheduler is None:
                raise RuntimeError("Diffusion variant requires a noise scheduler")
            noise = torch.randn_like(residual_target_norm)
            noise = self._apply_mask(noise, residual_valid_mask)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()
            noisy_residual = self.noise_scheduler.add_noise(residual_target_norm, noise, timesteps)
            noisy_residual = self._apply_mask(noisy_residual, residual_valid_mask)
            pred = self._predict_once(
                batch=batch,
                batch_size=batch_size,
                residual_norm=noisy_residual,
                timestep=timesteps,
                scene_memory=scene_memory,
            )
            pred = self._apply_mask(pred, residual_valid_mask)

            prediction_type = str(self.noise_scheduler.config.prediction_type)
            if prediction_type == "sample":
                target = residual_target_norm
                pred_residual_norm = pred
            elif prediction_type == "epsilon":
                target = noise
                self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].view(batch_size, 1)
                pred_residual_norm = (noisy_residual - torch.sqrt(1.0 - alpha_t) * pred) / torch.sqrt(alpha_t).clamp_min(1e-8)
            elif prediction_type == "v_prediction":
                self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
                self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
                alpha_t = self.noise_scheduler.alpha_t[timesteps].view(batch_size, 1)
                sigma_t = self.noise_scheduler.sigma_t[timesteps].view(batch_size, 1)
                target = alpha_t * noise - sigma_t * residual_target_norm
                pred_residual_norm = alpha_t * noisy_residual - sigma_t * pred
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")

            loss = self._masked_mse(pred, target, residual_valid_mask)
            loss_dict = {
                "model_loss": loss.detach(),
                "diffusion_loss": loss.detach(),
            }
        elif self.variant == "flow_matching":
            noise = torch.randn_like(residual_target_norm)
            noise = self._apply_mask(noise, residual_valid_mask)
            timesteps = self._sample_flow_matching_timesteps(batch_size, dtype=residual_target_norm.dtype)
            xt = timesteps.unsqueeze(-1) * noise + (1.0 - timesteps.unsqueeze(-1)) * residual_target_norm
            xt = self._apply_mask(xt, residual_valid_mask)
            flow_target = noise - residual_target_norm
            pred = self._predict_once(
                batch=batch,
                batch_size=batch_size,
                residual_norm=xt,
                timestep=timesteps,
                scene_memory=scene_memory,
            )
            pred = self._apply_mask(pred, residual_valid_mask)
            pred_residual_norm = self._apply_mask(noise - pred, residual_valid_mask)
            loss = self._masked_mse(pred, flow_target, residual_valid_mask)
            loss_dict = {
                "model_loss": loss.detach(),
                "flow_matching_loss": loss.detach(),
            }
        else:
            pred_residual_norm = self._predict_once(
                batch=batch,
                batch_size=batch_size,
                residual_norm=None,
                timestep=None,
                scene_memory=scene_memory,
            )
            pred_residual_norm = self._apply_mask(pred_residual_norm, residual_valid_mask)
            loss = self._masked_mse(pred_residual_norm, residual_target_norm, residual_valid_mask)
            loss_dict = {
                "model_loss": loss.detach(),
                "mlp_loss": loss.detach(),
            }

        if not return_aux:
            return loss, loss_dict

        pred_residual_norm = self._apply_mask(pred_residual_norm, residual_valid_mask)
        pred_residual = self._unnormalize_residual(pred_residual_norm, scale=res_scale, offset=res_offset)
        pred_residual = self._apply_mask(pred_residual, residual_valid_mask)
        pred_action = residual_to_absolute(pred_residual, q1_padded, slot=self._slot)
        pred_action = self._apply_action_valid_mask(pred_action, action_valid_mask)

        aux: Dict[str, torch.Tensor] = {
            "pred_action": pred_action,
            "action_valid_mask": action_valid_mask,
            "pred_residual_norm": pred_residual_norm,
            "residual_target_norm": residual_target_norm,
            "residual_valid_mask": residual_valid_mask,
        }
        return loss, loss_dict, aux

    # -------------------------------- inference --------------------------------------

    @torch.no_grad()
    def predict_action(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q1_padded = batch["q1_padded"].to(self.device, dtype=torch.float32)
        batch_size = int(q1_padded.shape[0])
        action_valid_mask = self._build_action_valid_mask(batch, batch_size=batch_size)
        residual_valid_mask = self._build_residual_valid_mask(batch, batch_size=batch_size)
        res_scale, res_offset = self._get_residual_normalizer_params(batch, batch_size=batch_size)
        # Scene tokens are a function of q1 and the contact cloud only, so we encode the scene
        # exactly once and reuse its memory across every denoising step.
        scene_memory = self._build_scene_memory(batch)

        if self.variant == "diffusion":
            if self.noise_scheduler is None:
                raise RuntimeError("Diffusion variant requires a noise scheduler")
            residual_norm = torch.randn((batch_size, self.residual_dim), device=self.device, dtype=torch.float32)
            residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for timestep in self.noise_scheduler.timesteps:
                model_output = self._predict_once(
                    batch=batch,
                    batch_size=batch_size,
                    residual_norm=residual_norm,
                    timestep=timestep,
                    scene_memory=scene_memory,
                )
                residual_norm = self.noise_scheduler.step(model_output, timestep, residual_norm).prev_sample
                residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
        elif self.variant == "flow_matching":
            residual_norm = torch.randn((batch_size, self.residual_dim), device=self.device, dtype=torch.float32)
            residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
            dt = 1.0 / float(self.num_inference_steps)
            for step_idx in range(self.num_inference_steps):
                t_value = 1.0 - float(step_idx) * dt
                timestep = torch.full((batch_size,), t_value, device=self.device, dtype=torch.float32)
                flow_pred = self._predict_once(
                    batch=batch,
                    batch_size=batch_size,
                    residual_norm=residual_norm,
                    timestep=timestep,
                    scene_memory=scene_memory,
                )
                residual_norm = residual_norm - dt * flow_pred
                residual_norm = self._apply_mask(residual_norm, residual_valid_mask)
        else:
            residual_norm = self._predict_once(
                batch=batch,
                batch_size=batch_size,
                residual_norm=None,
                timestep=None,
                scene_memory=scene_memory,
            )
            residual_norm = self._apply_mask(residual_norm, residual_valid_mask)

        residual = self._unnormalize_residual(residual_norm, scale=res_scale, offset=res_offset)
        residual = self._apply_mask(residual, residual_valid_mask)
        action_abs = residual_to_absolute(residual, q1_padded, slot=self._slot)
        action_abs = self._apply_action_valid_mask(action_abs, action_valid_mask)
        return {
            "action": action_abs,
            "action_pred": action_abs,
            "residual_pred": residual,
            "residual_norm_pred": residual_norm,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action(batch)
