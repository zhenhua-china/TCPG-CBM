import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import open_clip
import math
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Soft Top-K gating (可微、稳定)
# -------------------------
class SoftTopKGating(nn.Module):
    def __init__(self, topk: int = 4, tau: float = 0.1):
        super().__init__()
        self.topk = int(topk)
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (..., M)
        return weights: (..., M) with only topk kept (soft weights), renormalized.
        """
        w = torch.softmax(logits / self.tau, dim=-1)  # soft
        if self.topk <= 0 or self.topk >= w.size(-1):
            return w
        topv, topi = torch.topk(w, k=self.topk, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(-1, topi, topv)
        mask = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return mask


# -------------------------
# TCPA B-style injection layer
# - CLS token 用 CLS prompt pool
# - Patch tokens 用 Image prompt pool，并且每个 patch token 选 topK prompts
# -------------------------
class TCPAInjection(nn.Module):
    def __init__(
        self,
        dim: int,
        n_img_prompts: int = 20,
        n_cls_prompts: int = 4,
        topk: int = 4,
        tau: float = 0.1,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super().__init__()
        self.dim = dim

        # prompt tokens（可训练）
        self.img_prompts = nn.Parameter(torch.randn(n_img_prompts, dim) * 0.02)
        self.cls_prompts = nn.Parameter(torch.randn(n_cls_prompts, dim) * 0.02)

        # prompt indicators：用于 token->prompt 匹配（可训练）
        self.img_ind = nn.Parameter(torch.randn(n_img_prompts, dim) * 0.02)
        self.cls_ind = nn.Parameter(torch.randn(n_cls_prompts, dim) * 0.02)

        self.gate = SoftTopKGating(topk=topk, tau=tau)
        self.drop = nn.Dropout(dropout)

        # 残差缩放（训练更稳）
        self.scale_img = nn.Parameter(torch.tensor(1.0))
        self.scale_cls = nn.Parameter(torch.tensor(1.0))

        # 可选：融合后再过一个小 MLP（让增益更明显）
        self.use_mlp = use_mlp
        if use_mlp:
            self.ln = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1+P, D)
        return: (B, 1+P, D)
        """
        B, N, D = x.shape
        cls = x[:, :1, :]          # (B,1,D)
        patch = x[:, 1:, :]        # (B,P,D)

        # ---- 1) Patch tokens: token-coordinated image prompt attention (soft topK) ----
        # cosine logits: (B,P,M)
        q = F.normalize(patch, dim=-1)
        ind = F.normalize(self.img_ind, dim=-1)        # (M,D)
        logits = torch.einsum("bpd,md->bpm", q, ind)   # (B,P,M)

        w = self.gate(logits)                          # (B,P,M)
        # prompt value: (M,D) -> (B,P,D)  token-specific mixture
        delta_patch = torch.einsum("bpm,md->bpd", w, self.img_prompts)
        delta_patch = self.drop(delta_patch)

        patch = patch + self.scale_img * delta_patch

        # ---- 2) CLS token: dedicated CLS prompt attention ----
        q_cls = F.normalize(cls, dim=-1)               # (B,1,D)
        indc = F.normalize(self.cls_ind, dim=-1)       # (Mc,D)
        logits_cls = torch.einsum("bld,md->blm", q_cls, indc)  # (B,1,Mc)
        w_cls = torch.softmax(logits_cls, dim=-1)      # CLS 不必 topK，通常全用更稳
        delta_cls = torch.einsum("blm,md->bld", w_cls, self.cls_prompts)
        delta_cls = self.drop(delta_cls)

        cls = cls + self.scale_cls * delta_cls

        x = torch.cat([cls, patch], dim=1)

        # ---- 3) Optional MLP refinement (helps) ----
        if self.use_mlp:
            x = x + self.mlp(self.ln(x))

        return x


# -------------------------
# ViT Patch Extractor with TCPA injections
# - 支持插入所有层 or 最后K层 or 自定义层索引
# -------------------------
class ViT_PatchExtractor_TCPA(nn.Module):
    """
    ViT patch extractor + TCPA (scheme B-style) injections.
    Works with timm ViT without modifying timm source.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        # layer selection
        inject: str = "last_k",        # "all" / "last_k" / "custom"
        last_k: int = 4,
        custom_layers: Optional[List[int]] = None,  # e.g. [8,9,10,11]
        # TCPA params
        n_img_prompts: int = 20,
        n_cls_prompts: int = 4,
        topk: int = 4,
        tau: float = 0.1,
        dropout: float = 0.1,
        use_mlp: bool = True,
    ):
        super().__init__()
        import timm
        try:
            if weights_path:
                self.vit = timm.create_model(model_name, pretrained=False)
                sd = torch.load(weights_path, map_location='cpu')
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                self.vit.load_state_dict(sd, strict=False)
            else:
                self.vit = timm.create_model(model_name, pretrained=pretrained)
        except Exception as e:
            print(f"Warning: timm pretrained weights not available ({e}). Using random init.")
            self.vit = timm.create_model(model_name, pretrained=False)
        self.vit.head = nn.Identity()

        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.embed_dim = self.vit.embed_dim
        self.num_blocks = len(self.vit.blocks)

        # freeze backbone
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        # decide which blocks to inject
        inject = inject.lower()
        if inject == "all":
            self.inject_layers = list(range(self.num_blocks))
        elif inject == "last_k":
            k = max(0, min(int(last_k), self.num_blocks))
            self.inject_layers = list(range(self.num_blocks - k, self.num_blocks))
        elif inject == "custom":
            assert custom_layers is not None and len(custom_layers) > 0
            self.inject_layers = sorted([i for i in custom_layers if 0 <= i < self.num_blocks])
        else:
            raise ValueError("inject must be one of: all / last_k / custom")

        # one TCPAInjection per injected block
        self.tcpas = nn.ModuleDict()
        for i in self.inject_layers:
            self.tcpas[str(i)] = TCPAInjection(
                dim=self.embed_dim,
                n_img_prompts=n_img_prompts,
                n_cls_prompts=n_cls_prompts,
                topk=topk,
                tau=tau,
                dropout=dropout,
                use_mlp=use_mlp,
            )

        # ensure TCPA params trainable even if frozen backbone
        for p in self.tcpas.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, return_cls: bool = False):
        # patch embed -> (B,P,D)
        dev = None
        if hasattr(self.vit, "patch_embed") and hasattr(self.vit.patch_embed, "proj"):
            dev = self.vit.patch_embed.proj.weight.device
            dtype = self.vit.patch_embed.proj.weight.dtype
            x = x.to(dev, dtype=dtype, non_blocking=True)
        else:
            x = x.float()
        x = self.vit.patch_embed(x)
        B, P, D = x.shape

        # cls token
        if hasattr(self.vit, "cls_token"):
            cls = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls, x), dim=1)  # (B, 1+P, D)

        # pos embed
        if hasattr(self.vit, "pos_embed"):
            x = x + self.vit.pos_embed[:, : x.size(1), :]

        # pos drop (timm often has it)
        if hasattr(self.vit, "pos_drop"):
            x = self.vit.pos_drop(x)

        # transformer blocks + optional TCPA injection
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if str(i) in self.tcpas:
                x = self.tcpas[str(i)](x)

        x = self.vit.norm(x)

        if return_cls:
            return x[:, 1:, :], x[:, 0, :]
        else:
            return x[:, 1:, :]







# #
class ViT_PatchExtractor(nn.Module):
    """
    Extract patch features from images using Vision Transformer.

    Uses pre-trained ViT model (e.g., from timm library) to extract
    local patch features without the CLS token.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        freeze: bool = False
    ):
        """
        Args:
            model_name: Name of ViT model from timm
            weights_path: Optional local weights file path
            pretrained: Whether to use pre-trained weights
            freeze: Whether to freeze backbone parameters
        """
        super(ViT_PatchExtractor, self).__init__()

        try:
            import timm
            if weights_path:
                self.vit = timm.create_model(model_name, pretrained=False)
                import torch as _torch
                sd = _torch.load(weights_path, map_location='cpu')
                if isinstance(sd, dict) and 'state_dict' in sd:
                    sd = sd['state_dict']
                self.vit.load_state_dict(sd, strict=False)
            else:
                try:
                    self.vit = timm.create_model(model_name, pretrained=pretrained)
                except Exception as e:
                    print(f"Warning: timm pretrained weights not available ({e}). Using random init.")
                    self.vit = timm.create_model(model_name, pretrained=False)

            # Remove classification head
            self.vit.head = nn.Identity()

            # Optionally freeze backbone
            if freeze:
                for param in self.vit.parameters():
                    param.requires_grad = False

            self.available = True
            self.patch_size = self.vit.patch_embed.patch_size[0]
            self.embed_dim = self.vit.embed_dim

        except ImportError:
            print("Warning: timm not installed. Install with: pip install timm")
            self.available = False
            self.patch_size = 16
            self.embed_dim = 768

    def forward(self, x: torch.Tensor, return_cls: bool = False):

        # patch embed
        dev = None
        if hasattr(self.vit, "patch_embed") and hasattr(self.vit.patch_embed, "proj"):
            dev = self.vit.patch_embed.proj.weight.device
            dtype = self.vit.patch_embed.proj.weight.dtype
            x = x.to(dev, dtype=dtype, non_blocking=True)
        else:
            x = x.float()
        x = self.vit.patch_embed(x)  # [B,P,D]

        # ---- 加 CLS token ----
        if hasattr(self.vit, 'cls_token'):
            cls = self.vit.cls_token.expand(x.size(0), -1, -1)  # [B,1,D]
            x = torch.cat((cls, x), dim=1)  # [B,1+P,D]
        # ---- 加 position embedding（包含 CLS）----
        if hasattr(self.vit, 'pos_embed'):
            x = x + self.vit.pos_embed[:, :x.size(1), :]

        # blocks
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        if return_cls:
            cls_feat = x[:, 0, :]  # [B,D]
            patch_feat = x[:, 1:, :]  # [B,P,D]
            return patch_feat, cls_feat
        else:
            return x[:, 1:, :]  # 仍旧返回 patch，兼容你原逻辑







class CLIP_ConceptEncoder(nn.Module):
    """
    Encode text concepts using CLIP text encoder.
    
    Concepts are encoded once and cached for efficiency.
    """
    
    def __init__(
        self,
        concept_list: List[str],
        model_name: str = 'ViT-B-32',
        device: str = 'cuda',
        pretrained_path: str = 'F:/OTCBM-main/open_clip_pytorch_model.bin'
    ):
        super(CLIP_ConceptEncoder, self).__init__()
        
        self.concept_list = concept_list
        self.device = device
        if not pretrained_path:
            raise ValueError("pretrained_path is required for CLIP_ConceptEncoder")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None
        )
        self.clip_model = self.clip_model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        sd = torch.load(pretrained_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        self.clip_model.load_state_dict(sd, strict=False)
        with torch.no_grad():
            self.concept_features = self._encode_concepts(concept_list)
        self.embed_dim = self.concept_features.shape[-1]
    
    def _encode_concepts(self, concepts: List[str]) -> torch.Tensor:
        """
        Encode concept texts to feature vectors.
        
        Args:
            concepts: List of concept names
            
        Returns:
            concept_features: [num_concepts, embed_dim]
        """
        # Create text prompts
        texts = [f"a photo of {concept}" for concept in concepts]
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # Encode texts
        text_features = self.clip_model.encode_text(text_tokens)
        
        # Normalize to unit sphere
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get concept features for a batch.
        
        Args:
            batch_size: Number of samples in batch
            
        Returns:
            concept_features: [batch_size, num_concepts, embed_dim]
        """
        if self.concept_features is None:
            num_concepts = len(self.concept_list)
            return torch.randn(batch_size, num_concepts, self.embed_dim, device=self.device)
        
        # Expand to batch dimension
        return self.concept_features.unsqueeze(0).expand(batch_size, -1, -1)
    
    # def update_concepts(self, new_concepts: List[str]):
    #     """
    #     Update concept list and re-encode.
    #
    #     Args:
    #         new_concepts: New list of concept names
    #     """
    #     if self.available:
    #         self.concept_list = new_concepts
    #         with torch.no_grad():
    #             self.concept_features = self._encode_concepts(new_concepts)
