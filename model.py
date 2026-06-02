import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from importlib import import_module
import math
from utils.presence_loss import presence_loss_tanh

 



class PartDiscoveryHead(nn.Module):
    def __init__(self, num_parts: int, hidden_dim: int, tv_weight: float = 0.1, distinct_weight: float = 0.1, presence_weight: float = 0.1):
        super().__init__()
        self.K = int(num_parts)
        self.D = int(hidden_dim)
        self.prototypes = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.fg_vector = nn.Parameter(torch.randn(self.D) * 0.02)
        self.tv_weight = float(tv_weight)
        self.distinct_weight = float(distinct_weight)
        self.presence_weight = float(presence_weight)

    def _tv_loss(self, w_maps: torch.Tensor) -> torch.Tensor:
        dh = (w_maps[:, :, :, 1:] - w_maps[:, :, :, :-1]).abs().mean()
        dv = (w_maps[:, :, 1:, :] - w_maps[:, :, :-1, :]).abs().mean()
        return dh + dv

    def _distinct_loss(self, w: torch.Tensor) -> torch.Tensor:
        B, K, P = w.shape
        g = torch.bmm(w, w.transpose(1, 2))
        mask = ~torch.eye(K, dtype=torch.bool, device=w.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return g[mask].mean()

    def forward(self, patch_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, D = patch_features.shape
        q = F.normalize(self.prototypes, dim=-1)
        x = F.normalize(patch_features, dim=-1)
        logits = torch.einsum("bpd,kd->bpk", x, q)  # [B,P,K]
        a = torch.softmax(logits, dim=2)            # per-patch parts mixture [B,P,K]
        s = torch.sigmoid(torch.einsum("bpd,d->bp", patch_features, self.fg_vector))  # [B,P]
        m_parts = a * s.unsqueeze(2)                # foreground gating
        w = m_parts.transpose(1, 2)                 # [B,K,P]
        w_norm = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        z = torch.bmm(w_norm, patch_features)       # [B,K,D]
        g = int(math.sqrt(P))
        if g * g == P:
            maps_parts = m_parts.view(B, P, self.K).permute(0, 2, 1).contiguous().view(B, self.K, g, g)
            tv = self._tv_loss(maps_parts)
            prs = presence_loss_tanh(maps_parts)
            m_bg = (1.0 - s).view(B, 1, g, g)
            maps = torch.cat([maps_parts, m_bg], dim=1)  # [B,K+1,g,g]
        else:
            tv = torch.tensor(0.0, device=patch_features.device)
            prs = torch.tensor(0.0, device=patch_features.device)
            maps = torch.zeros(B, self.K + 1, 1, 1, device=patch_features.device)
        dst = self._distinct_loss(w)
        reg = self.tv_weight * tv + self.distinct_weight * dst + self.presence_weight * prs
        return z, w_norm, s, reg, maps


class DOT_CBM(nn.Module):
    """
    Disentangled Optimal Transport Concept Bottleneck Model
    
    optimal transport between image patches and textual concepts.
    
    Args:
        num_patches (int): Number of image patches (e.g., 196 for ViT-B/16)
        num_concepts (int): Number of pre-defined concepts
        num_classes (int): Number of output classes
        patch_dim (int): Dimension of patch features
        concept_dim (int): Dimension of concept features
        hidden_dim (int): Hidden dimension for adapters
        ot_reg (float): Sinkhorn regularization parameter
        ot_max_iter (int): Maximum iterations for Sinkhorn algorithm
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        num_patches: int = 196,
        num_concepts: int = 312,
        num_classes: int = 200,
        patch_dim: int = 768,
        concept_dim: int = 512,
        hidden_dim: int = 256,
        ot_reg: float = 0.2,
        ot_max_iter: int = 100,
        dropout: float = 0.1,
        activation_method: str = "noisy_or",
        activation_tau: float = 1.0,
        use_parts: bool = False,
        num_parts: int = 8,
        part_tv_weight: float = 0.1,
        part_distinct_weight: float = 0.1,
        part_presence_weight: float = 0.1,
        concept_head: str = "nam"
    ):
        super(DOT_CBM, self).__init__()
        
        self.num_patches = num_patches
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.ot_reg = ot_reg
        self.ot_max_iter = ot_max_iter
        self.activation_method = activation_method
        self.activation_tau = activation_tau
        self.act_calib_scale = nn.Parameter(torch.ones(self.num_concepts))
        self.act_calib_bias = nn.Parameter(torch.zeros(self.num_concepts))
        self.use_parts = bool(use_parts)
        self.num_parts = int(num_parts)
        self.concept_head_type = concept_head
        
        # Visual adapter: maps patch features to alignment space
        self.visual_adapter = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Text adapter: maps concept features to alignment space
        self.text_adapter = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Concept to class classifier
        if self.concept_head_type.lower() == "mlp":
            self.concept_to_class = nn.Sequential(
                nn.Linear(num_concepts, num_classes * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_classes * 2, num_classes)
            )
        else:
            concept_head_module = import_module("class")
            concept_head_cls = getattr(concept_head_module, "ConceptToClass_NAM_InteractionHead")
            self.concept_to_class = concept_head_cls(num_concepts=num_concepts, num_classes=num_classes)

        if self.use_parts:
            self.part_head = PartDiscoveryHead(
                num_parts=self.num_parts,
                hidden_dim=hidden_dim,
                tv_weight=part_tv_weight,
                distinct_weight=part_distinct_weight,
                presence_weight=part_presence_weight
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_orthogonal_loss(
            self,
            features: torch.Tensor,
            weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Orthogonal loss with optional saliency weights (for VOP).

        Args:
            features: [B, N, D]
            weights:  [B, N]  (optional) patch saliency/importance. If provided,
                      off-diagonal similarities are weighted by w_i * w_j.
        """
        B, N, D = features.shape

        # Normalize features
        x = F.normalize(features, dim=-1)

        # Similarity matrix
        sim = torch.bmm(x, x.transpose(1, 2))  # [B,N,N]

        # mask off-diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)

        # original style: penalize squared off-diagonal similarities
        sim_sq = sim ** 2

        if weights is not None:
            # weights: [B,N] -> pairwise weights [B,N,N]
            w = weights.unsqueeze(2) * weights.unsqueeze(1)  # [B,N,N]
            loss = (sim_sq[mask] * w[mask]).mean()
        else:
            loss = sim_sq[mask].mean()

        return loss

    def solve_optimal_transport(
        self,
        patch_features: torch.Tensor,
        concept_features: torch.Tensor,
        visual_prior: Optional[torch.Tensor] = None,
        concept_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve optimal transport between patches and concepts using Sinkhorn algorithm.
        
        The OT problem is:
            min_{T} <T, C> + ε·H(T)
            s.t. T·1 = a, T^T·1 = b
        
        Args:
            patch_features: [batch, num_patches, dim] adapted patch features
            concept_features: [batch, num_concepts, dim] adapted concept features
            visual_prior: [batch, num_patches] optional prior distribution over patches
            concept_prior: [batch, num_concepts] optional prior distribution over concepts
            
        Returns:
            assignment: [batch, num_patches, num_concepts] optimal transport plan
            cost: [batch, num_patches, num_concepts] cost matrix
        """
        batch_size = patch_features.shape[0]
        device = patch_features.device
        
        # Normalize features to unit sphere for cosine similarity
        patch_norm = F.normalize(patch_features, dim=-1)
        concept_norm = F.normalize(concept_features, dim=-1)
        
        # Compute non-negative OT cost using 1 - cosine similarity
        # Higher similarity -> lower cost in [0, 2]
        similarity = torch.bmm(patch_norm, concept_norm.transpose(1, 2))
        cost = 1.0 - similarity
        # cost: [batch, num_patches, num_concepts]
        
        # Set default uniform priors if not provided
        if visual_prior is None:
            visual_prior = torch.ones(
                batch_size, patch_features.shape[1], device=device
            ) / patch_features.shape[1]
        
        if concept_prior is None:
            concept_prior = torch.ones(
                batch_size, concept_features.shape[1], device=device
            ) / concept_features.shape[1]
        
        # Sinkhorn scaling (PyTorch) to satisfy both marginals without POT
        # Kernel K = exp(-cost / eps), then T = diag(u) K diag(v) with T·1=a and T^T·1=b
        eps = self.ot_reg
        K = torch.exp(-cost / eps)  # [B, P, C]
        a = visual_prior  # [B, P]
        b = concept_prior  # [B, C]
        # Initialize scalings
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        # Iterative scaling
        for _ in range(self.ot_max_iter):
            Kv = torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-12  # [B, P]
            u = a / Kv
            KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-12  # [B, C]
            v = b / KTu
        assignment = u.unsqueeze(2) * K * v.unsqueeze(1)  # [B, P, C]
        
        return assignment, cost
    #
    def compute_concept_activation(
        self,
        assignment: torch.Tensor,
        cost: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute concept activation values from transport assignment.

        The activation of each concept is computed by aggregating
        the transport weights from all patches, weighted by similarity.

        Args:
            assignment: [batch, num_patches, num_concepts] transport plan
            cost: [batch, num_patches, num_concepts] cost matrix

        Returns:
            concept_activation: [batch, num_concepts]
        """
        sim_weight = torch.exp(-cost / self.activation_tau)
        weighted = assignment * sim_weight
        if self.activation_method == "weighted_mean":
            denom = assignment.sum(dim=1) + 1e-8
            concept_activation = weighted.sum(dim=1) / denom
        else:
            sim_weight = torch.exp(-cost)
            evid = assignment * sim_weight

            k = min(getattr(self, "activation_topk", 8), evid.size(1))
            topv, topi = torch.topk(evid, k=k, dim=1)

            ass_topk = torch.gather(assignment, dim=1, index=topi)
            numer = topv.sum(dim=1)
            denom = ass_topk.sum(dim=1) + 1e-8
            act = numer / denom

            eps = 1e-6
            act = act.clamp(eps, 1.0 - eps)
            logit_act = torch.log(act) - torch.log(1.0 - act)
            concept_activation = torch.sigmoid(logit_act * self.act_calib_scale + self.act_calib_bias)

        return concept_activation


    
    def forward(
        self,
        patch_features: torch.Tensor,
        concept_features: torch.Tensor,
        visual_prior: Optional[torch.Tensor] = None,
        concept_prior: Optional[torch.Tensor] = None,
        return_assignment: bool = False
    ) -> Tuple:
        """
        Forward pass of DOT-CBM.
        
        Args:
            patch_features: [batch, num_patches, patch_dim]
            concept_features: [batch, num_concepts, concept_dim]
            visual_prior: [batch, num_patches] optional visual prior
            concept_prior: [batch, num_concepts] optional concept prior
            return_assignment: whether to return OT assignment matrix
            
        Returns:
            class_logits: [batch, num_classes] class predictions
            concept_activation: [batch, num_concepts] concept activations
            orth_loss: scalar tensor, orthogonality loss
            assignment (optional): [batch, num_patches, num_concepts] OT plan
        """
        # Step 1: Apply adapters to map features to alignment space
        patch_adapted = self.visual_adapter(patch_features)
        concept_adapted = self.text_adapter(concept_features)
        part_reg = torch.tensor(0.0, device=patch_features.device)
        if self.use_parts:
            z_parts, w_parts, s_fg, reg, _maps = self.part_head(patch_adapted)
            patch_adapted = z_parts
            part_reg = reg
        
        # Step 2: Compute orthogonal losses for disentanglement
        orth_loss_v = self.compute_orthogonal_loss(patch_adapted)
        orth_loss_t = self.compute_orthogonal_loss(concept_adapted)
        orth_loss = orth_loss_v + orth_loss_t + part_reg
        
        # Step 3: Solve optimal transport between patches and concepts
        assignment, cost = self.solve_optimal_transport(
            patch_adapted, concept_adapted,
            visual_prior, concept_prior
        )
        
        # Step 4: Compute concept activation values
        concept_activation = self.compute_concept_activation(assignment, cost)
        
        # Step 5: Predict class from concepts
        class_logits = self.concept_to_class(concept_activation)
        
        if return_assignment:
            return class_logits, concept_activation, orth_loss, assignment
        else:
            return class_logits, concept_activation, orth_loss
    
    def get_parts(
        self,
        patch_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.visual_adapter(patch_features)
        if not self.use_parts:
            B, P, D = patch_features.shape
            return patch_features.new_zeros(B, 0, D), patch_features.new_zeros(B, 0, P), patch_features.new_zeros(B, P), patch_features.new_zeros(B, 0, 1, 1)
        z_parts, w_parts, s_fg, _reg, maps = self.part_head(x)
        return z_parts, w_parts, s_fg, maps
    
    @torch.no_grad()
    def get_assignment_and_cost(
        self,
        patch_features: torch.Tensor,
        concept_features: torch.Tensor,
        visual_prior: Optional[torch.Tensor] = None,
        concept_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute OT assignment and cost given raw features.
        """
        patch_adapted = self.visual_adapter(patch_features)
        concept_adapted = self.text_adapter(concept_features)
        assignment, cost = self.solve_optimal_transport(
            patch_adapted, concept_adapted, visual_prior, concept_prior
        )
        return assignment, cost
    
    def get_concept_weights(self) -> torch.Tensor:
        """
        Get the learned weights from concepts to classes.
        This can be used for interpretability analysis.
        
        Returns:
            weights: if FAME classifier, [num_concepts, num_rules, num_classes], else [num_concepts, num_classes]
        """
        if isinstance(self.concept_to_class):
            return torch.stack([m.consequent for m in self.concept_to_class.sf_ls], dim=0)
        if hasattr(self.concept_to_class, "w2"):
            return self.concept_to_class.w2.mean(dim=1)
        return None


if __name__ == '__main__':
    # Test the model
    print("Testing DOT-CBM model...")
    
    # Create model
    model = DOT_CBM(
        num_patches=196,
        num_concepts=312,
        num_classes=200,
        patch_dim=768,
        concept_dim=512
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 4
    patch_features = torch.randn(batch_size, 196, 768)
    concept_features = torch.randn(batch_size, 312, 512)
    
    # Forward pass
    class_logits, concept_activation, orth_loss, assignment = model(
        patch_features, concept_features, return_assignment=True
    )
    
    print(f"\nForward pass successful!")
    print(f"Class logits shape: {class_logits.shape}")
    print(f"Concept activation shape: {concept_activation.shape}")
    print(f"Assignment shape: {assignment.shape}")
    print(f"Orthogonal loss: {orth_loss.item():.4f}")
    
    # Test gradient flow
    loss = class_logits.sum() + concept_activation.sum() + orth_loss
    loss.backward()
    print(f"\nBackward pass successful!")
    
    print("\n✓ Model test passed!")
