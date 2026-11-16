import math
import numpy as np
import torch


def wrap_angle(theta):
    return theta % math.pi


def box_corners(box):
    cx, cy, w, h, angle = map(float, box)
    offsets = np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
    rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    return offsets @ rotation.T + [cx,cy]


def polygon_area(polygon):
    p = np.asarray(polygon)
    if len(p) < 3:
        return 0.0
    return float(abs(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1)))/2)


def polygon_intersection(subject, clip):
    output = [np.asarray(p,dtype=float) for p in subject]
    cross = lambda a,b: a[0]*b[1]-a[1]*b[0]
    for a,b in zip(clip, np.roll(clip,-1,axis=0)):
        source, output = output, []
        if not source:
            break
        def inside(p):
            return cross(b-a,p-a) >= -1e-10
        def intersect(p,q):
            direction = q-p
            den = cross(direction,b-a)
            if abs(den) < 1e-15:
                return q
            return p + cross(a-p,b-a)/den*direction
        p = source[-1]
        for q in source:
            if inside(q):
                if not inside(p): output.append(intersect(p,q))
                output.append(q)
            elif inside(p):
                output.append(intersect(p,q))
            p = q
    return output


def rotated_iou(a,b):
    pa,pb=box_corners(a),box_corners(b)
    overlap=polygon_area(polygon_intersection(pa,pb))
    return overlap/max(float(a[2]*a[3]+b[2]*b[3])-overlap,1e-12)


def rotated_nms(boxes,scores,labels,threshold=.5,max_detections=300):
    boxes,scores,labels=np.asarray(boxes),np.asarray(scores),np.asarray(labels)
    order=np.lexsort((np.arange(len(scores)),labels,-scores))
    keep=[]
    for i in order:
        if all(labels[i]!=labels[j] or rotated_iou(boxes[i],boxes[j]) <= threshold for j in keep):
            keep.append(int(i))
            if len(keep)>=max_detections:break
    return keep


def encode_boxes(boxes,image_size):
    width,height=image_size
    scale=boxes.new_tensor([width,height,width,height])
    first=boxes[...,:4]/scale
    return torch.cat((first[...,:2],first[...,2:].clamp_min(1e-8).log(),
                      torch.cos(2*boxes[...,4:5]),torch.sin(2*boxes[...,4:5])), -1)


def decode_boxes(code,image_size):
    width,height=image_size
    scale=code.new_tensor([width,height,width,height])
    first=torch.cat((code[...,:2],code[...,2:4].clamp(-12,4).exp()),-1)*scale
    angle=.5*torch.atan2(code[...,5],code[...,4]) % math.pi
    return torch.cat((first,angle.unsqueeze(-1)),-1)


def points_in_boxes(points,boxes,center_fraction=1.0):
    delta=points[:,None,:]-boxes[None,:,:2]
    c,s=boxes[:,4].cos(),boxes[:,4].sin()
    dx=delta[...,0]*c+delta[...,1]*s
    dy=-delta[...,0]*s+delta[...,1]*c
    return (dx.abs()<=boxes[None,:,2]*center_fraction/2)&(dy.abs()<=boxes[None,:,3]*center_fraction/2)
