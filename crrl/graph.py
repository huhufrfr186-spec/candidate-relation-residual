from dataclasses import dataclass
import torch
from torch import nn


def reduced_incidence(edges, n_left, n_right):
    edges = torch.as_tensor(edges, dtype=torch.long)
    parent = list(range(n_left + n_right))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    active = set()
    for i, j in edges.tolist():
        j += n_left
        active.update((i, j))
        parent[find(j)] = find(i)
    anchors = {}
    for i in sorted(active):
        anchors.setdefault(find(i), i)
    keep = [i for i in sorted(active) if i not in anchors.values()]
    full = torch.zeros(len(edges), n_left + n_right, dtype=torch.float64)
    if len(edges):
        full[torch.arange(len(edges)), edges[:, 0]] = 1
        full[torch.arange(len(edges)), n_left + edges[:, 1]] = 1
    return full, full[:, keep], len(anchors)


def weighted_projection(features, mass, design, eta=0.05):
    original_dtype = features.dtype
    h, p = features.double(), mass.double()
    b = design.to(device=h.device, dtype=torch.float64)
    if h.shape[-2] == 0:
        return features
    if b.shape[-1] == 0:
        return features
    w = eta + (1 - eta) * p
    lhs = b.T @ (w.unsqueeze(-1) * b)
    rhs = b.T @ (w.unsqueeze(-1) * h)
    solution = torch.linalg.solve(lhs, rhs)
    return (h - b @ solution).to(original_dtype)


def _transport_normal(plan):
    cross=plan[...,:,:-1]
    top=torch.cat((torch.diag_embed(plan.sum(-1)),cross),-1)
    bottom=torch.cat((cross.transpose(-1,-2),torch.diag_embed(plan.sum(-2)[...,:-1])),-1)
    return torch.cat((top,bottom),-2)


class _EntropicTransport(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,support,unmatched,temperature,iterations):
        n,m=scores.shape[-2:];s=scores.double()/temperature
        dust=(unmatched.double()/temperature).expand(*s.shape[:-2],1,1)
        real=s.masked_fill(~support.to(s.device),-torch.inf)
        top=torch.cat((real,dust.expand(*s.shape[:-2],n,1)),-1)
        z=torch.cat((top,dust.expand(*s.shape[:-2],1,m+1)),-2)
        row=s.new_ones(n+1);row[-1]=m;col=s.new_ones(m+1);col[-1]=n
        u,v=torch.zeros_like(z[..., :,0]),torch.zeros_like(z[...,0,:])
        for index in range(iterations):
            u=row.log()-torch.logsumexp(z+v.unsqueeze(-2),-1)
            v=col.log()-torch.logsumexp(z+u.unsqueeze(-1),-2)
            if (index+1)%20==0:
                plan=(z+u.unsqueeze(-1)+v.unsqueeze(-2)).exp()
                if (plan.sum(-1)-row).abs().max()<1e-10:break
        # Dual Newton refinement enforces marginal accuracy for sharp plans.
        u=u+v[...,-1:];v=v-v[...,-1:]
        for _ in range(30):
            plan=(z+u.unsqueeze(-1)+v.unsqueeze(-2)).exp()
            error=torch.cat((plan.sum(-1)-row,(plan.sum(-2)-col)[...,:-1]),-1)
            norm=error.abs().amax(-1)
            if norm.max()<1e-10:break
            step=torch.linalg.solve(_transport_normal(plan),error.unsqueeze(-1)).squeeze(-1)
            du=step[...,:n+1];dv=torch.cat((step[...,n+1:],step.new_zeros(*step.shape[:-1],1)),-1)
            alpha=torch.ones_like(norm)
            for _ in range(30):
                candidate=(z+(u-alpha[...,None]*du).unsqueeze(-1)+(v-alpha[...,None]*dv).unsqueeze(-2)).exp()
                candidate_error=torch.cat((candidate.sum(-1)-row,(candidate.sum(-2)-col)[...,:-1]),-1).abs().amax(-1)
                accept=torch.isfinite(candidate_error)&((candidate_error<norm)|(norm<1e-10))
                if accept.all():break
                alpha=torch.where(accept,alpha,alpha*.5)
            u=u-alpha[...,None]*du;v=v-alpha[...,None]*dv
        plan=(z+u.unsqueeze(-1)+v.unsqueeze(-2)).exp()
        residual=max(float((plan.sum(-1)-row).abs().max()),float((plan.sum(-2)-col).abs().max()))
        if not torch.isfinite(plan).all() or residual>1e-7:
            raise RuntimeError(f'Transport marginal solver did not converge: residual={residual:.3g}')
        ctx.save_for_backward(plan);ctx.temperature=temperature;ctx.dtype=scores.dtype;ctx.unmatched_dtype=unmatched.dtype
        return plan

    @staticmethod
    def backward(ctx,gradient):
        plan,=ctx.saved_tensors
        weighted=plan*gradient.double()
        rhs=torch.cat((weighted.sum(-1),weighted.sum(-2)[...,:-1]),-1)
        potential=torch.linalg.solve(_transport_normal(plan),rhs.unsqueeze(-1)).squeeze(-1)
        rows=plan.shape[-2]
        a=potential[...,:rows];b=torch.cat((potential[...,rows:],potential.new_zeros(*potential.shape[:-1],1)),-1)
        derivative=plan*(gradient.double()-a.unsqueeze(-1)-b.unsqueeze(-2))/ctx.temperature
        score_grad=derivative[...,:-1,:-1].to(ctx.dtype)
        unmatched_grad=(derivative[...,:-1,-1].sum()+derivative[...,-1,:].sum()).to(ctx.unmatched_dtype)
        return score_grad,None,unmatched_grad,None,None


def soft_transport(scores, support, unmatched, temperature=0.1, iterations=100):
    """Entropy-optimal augmented transport with implicit KKT differentiation."""
    n,m=scores.shape[-2:]
    if temperature<=0 or iterations<1:raise ValueError('Positive temperature and iteration count are required')
    if n==0 or m==0 or not support.any():
        out=scores.new_zeros(*scores.shape[:-2],n+1,m+1)
        if n:out[...,:n,m]=1
        if m:out[...,n,:m]=1
        return out
    return _EntropicTransport.apply(scores,support,unmatched,temperature,iterations)


class BoundedCorrection(nn.Module):
    def __init__(self, channels, outputs, beta, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, outputs))
        amplitude = torch.as_tensor(beta, dtype=torch.float32)
        if amplitude.numel() not in (1, outputs) or (amplitude < 0).any():
            raise ValueError("beta must be nonnegative and scalar or output-sized")
        self.register_buffer("beta", amplitude)

    def forward(self, residual):
        return self.beta * torch.tanh(self.net(residual) - self.net(torch.zeros_like(residual)))


@dataclass
class GraphGroup:
    queries: torch.Tensor
    candidates: torch.Tensor
    support: torch.Tensor
    edges: torch.Tensor
    design: torch.Tensor
    relative: torch.Tensor


def grid_points(height, width, device=None, dtype=torch.float32):
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack(((xx.flatten() + .5) / width, (yy.flatten() + .5) / height), -1).to(dtype)


def local_graphs(height, width, block=2, radius=2):
    if min(height,width,block)<1 or radius<0:
        raise ValueError('Grid and block dimensions must be positive; radius must be nonnegative')
    groups = {}
    points = grid_points(height, width)
    for y in range(0, height, block):
        for x in range(0, width, block):
            qs = [(v, u) for v in range(y, min(y + block, height)) for u in range(x, min(x + block, width))]
            candidates = sorted({(v, u) for a, b in qs
                                 for v in range(max(0, a-radius), min(height, a+radius+1))
                                 for u in range(max(0, b-radius), min(width, b+radius+1))})
            support = torch.tensor([[abs(a-v) <= radius and abs(b-u) <= radius for v,u in candidates] for a,b in qs])
            key = (len(qs), len(candidates), bytes(support.flatten().tolist()))
            entry = groups.setdefault(key, {"qs": [], "cs": [], "support": support})
            entry["qs"].append([a*width+b for a,b in qs])
            entry["cs"].append([a*width+b for a,b in candidates])
    out = []
    for entry in groups.values():
        qs, cs, support = torch.tensor(entry["qs"]), torch.tensor(entry["cs"]), entry["support"]
        edges = support.nonzero()
        _, design, _ = reduced_incidence(edges, support.shape[0], support.shape[1])
        rel = points[cs[:, edges[:,1]]] - points[qs[:, edges[:,0]]]
        out.append(GraphGroup(qs, cs, support, edges, design, rel))
    return out
