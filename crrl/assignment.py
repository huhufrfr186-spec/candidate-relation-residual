import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .geometry import points_in_boxes


def lexicographic_assignment(cost, tolerance=1e-12):
    """Minimum cost injection, choosing the lexicographically first optimum."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n > m:
        raise ValueError(f"Query capacity {m} is below the object count {n}")
    if n == 0:
        return np.empty(0,dtype=np.int64)
    rows, current = linear_sum_assignment(cost)
    optimum = float(cost[rows,current].sum())
    selected, available, fixed = [], list(range(m)), 0.0
    for i in range(n):
        remaining = cost[i:,available]
        rr, cc = linear_sum_assignment(remaining)
        incumbent = available[int(cc[0])]
        chosen = incumbent
        for candidate in available:
            if candidate >= incumbent:
                break
            pool = [c for c in available if c != candidate]
            lower = fixed + cost[i,candidate]
            if i+1 < n:
                lower += cost[i+1:,pool].min(axis=1).sum()
            if lower > optimum + tolerance:
                continue
            total = fixed + cost[i,candidate]
            if i+1 < n:
                rr, cc = linear_sum_assignment(cost[i+1:,pool])
                total += cost[i+1:,pool][rr,cc].sum()
            if abs(total-optimum) <= tolerance:
                chosen = candidate
                break
        selected.append(chosen)
        fixed += cost[i,chosen]
        available.remove(chosen)
    return np.asarray(selected,dtype=np.int64)


def assign_targets(points, boxes, image_size, center_fraction=.5, rule="unique"):
    owner = torch.full((len(points),),-1,dtype=torch.long,device=points.device)
    if len(boxes)==0:
        return owner
    width,height=image_size
    scale=points.new_tensor([width,height])
    if rule == "unique":
        cost=((boxes[:,:2,None]/scale[:,None])-points.T[None]).square().sum(1)
        reserved=lexicographic_assignment(cost.detach().cpu().numpy())
        owner[torch.as_tensor(reserved,device=owner.device)] = torch.arange(len(boxes),device=owner.device)
    elif rule not in ("center", "nearest"):
        raise ValueError(f"Unknown assignment rule: {rule}")
    if rule == "nearest":
        distance=torch.cdist(points, boxes[:,:2]/scale)
        selected=distance.argmin(0)
        for j,q in enumerate(selected.tolist()):
            if owner[q]<0 or boxes[j,2:4].prod()<boxes[owner[q],2:4].prod():owner[q]=j
    inside=points_in_boxes(points*scale,boxes,center_fraction)
    area=boxes[:,2:4].prod(-1)
    candidates=area.expand(len(points),-1).clone().masked_fill(~inside,float("inf"))
    smallest=candidates.argmin(-1)
    mask=(owner<0)&inside.any(-1)
    owner[mask]=smallest[mask]
    return owner
