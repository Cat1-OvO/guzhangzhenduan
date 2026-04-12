import torch
import torch.nn.functional as F

def instance_contrastive_loss(z1, z2, temperature=0.5):
    # 维度保护
    if z1.dim() == 3:
        z1 = F.adaptive_avg_pool1d(z1, 1).squeeze(-1)
        z2 = F.adaptive_avg_pool1d(z2, 1).squeeze(-1)
    elif z1.dim() > 2:
        z1 = z1.view(z1.size(0), -1)
        z2 = z2.view(z2.size(0), -1)

    # 关键修复：L2 normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)   # [2B, D]
    sim = torch.matmul(z, z.transpose(0, 1)) / temperature

    logits = torch.tril(sim, diagonal=-1)[:, :-1]
    logits += torch.triu(sim, diagonal=1)[:, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[i, B + i - 1].mean() + logits[B + i, i].mean()) / 2
    return loss