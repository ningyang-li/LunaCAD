import torch.nn as nn
import torch
import torch.nn.functional as F


class HGC(nn.Module):
    """
    Hierarchical Semantic Consistence (HGC) module.
    use soft Wasserstein distance to restrict the DBs of adjacent encoders to ensuer the semantic consistency
    """
    def __init__(self, num_layers=6, dim=256, tau=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.tau = tau
        # cross layer projection (l -> l+1)
        self.cross_proj = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers - 1)
        ])
        for proj in self.cross_proj:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0.0)

    def compute_region_descriptors(self, features, att):
        """
        Describe distribution balls with features and att
        features: (B, N, D)  —— output encoder features
        att: (B, N, K)       —— pixel-ball weights
        return: (K, D)       —— average descriptors
        """
        # the weights of each ball for all pixels (B, K)
        weight = att.sum(dim=1) + 1e-6
        # weighted aggregationn: (B, N, D) x (B, N, K) -> (B, K, D)
        features = F.normalize(features, p=2, dim=2)   # <-- ESSENTIAL, OR MODEL MAYBE NOT CONVERGENCE
        descriptors = torch.einsum('bnd,bnk->bkd', features, att)
        descriptors = descriptors / (weight.unsqueeze(-1) + 1e-6)  # (B, K, D)
        
        return descriptors.mean(dim=0)  # (K, D)

    def wasserstein_soft(self, src, tgt):
        """
        Wasserstein-2 based on center distance
        src: (K1, D), tgt: (K2, D)
        """
        cost = torch.cdist(src, tgt, p=2) ** 2   # (K1, K2)
        T = F.softmax(-cost / self.tau, dim=1)   # soft gradient (K1, K2)
        loss = (T * cost).sum()
        return loss

    def forward(self, layer_centers, layer_features, layer_attentions):
        """
        layer_centers:     list of (K_l, D)
        layer_features:    list of (B, N_l, D)
        layer_attentions:  list of (B, N_l, K_l)
        """
        if len(layer_centers) < 2:
            return torch.tensor(0.0, device=layer_centers[0].device)
        
        total_loss = 0
        for l in range(len(layer_centers) - 1):
            # DB descriptor of l-th layer
            desc_l = self.compute_region_descriptors(layer_features[l], layer_attentions[l]
            )  # (K_l, D)
            # Project to l+1-th space
            desc_l_proj = self.cross_proj[l](desc_l)  # (K_l, D)
            # DB descriptor of l+1-th layer
            desc_l1 = self.compute_region_descriptors(layer_features[l+1], layer_attentions[l+1])  # (K_{l+1}, D)
            total_loss += self.wasserstein_soft(desc_l_proj, desc_l1)

        loss = total_loss / (len(layer_centers) - 1)
        
        return loss if not torch.isnan(loss) else torch.tensor(0., device=loss.device)



        