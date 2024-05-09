import time
from functools import partial, reduce

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool3d

from .conv_backbone import convnext_3d_small, convnext_3d_tiny, convnextv2_3d_pico, convnextv2_3d_femto, clip_vitL14
from .head import IQAHead, VARHead, VQAHead
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_small, swin_3d_tiny


class BaseEvaluator(nn.Module):
    def __init__(
        self, backbone=dict(), vqa_head=dict(),
    ):
        super().__init__()
        self.backbone = VideoBackbone(**backbone)
        self.vqa_head = VQAHead(**vqa_head)

    def forward(self, vclip, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class COVER(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys="fragments,resize",
        multi=False,
        layer=-1,
        backbone=dict(
            resize={"window_size": (4, 4, 4)}, fragments={"window_size": (4, 4, 4)}
        ),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()
        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == "swin_tiny":
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == "swin_tiny_grpb":
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == "swin_tiny_grpb_m":
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == "swin_small":
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == "conv_tiny":
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == "conv_small":
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == "conv_femto":
                b = convnextv2_3d_femto(pretrained=True)
            elif t_backbone_size == "conv_pico":
                b = convnextv2_3d_pico(pretrained=True)
            elif t_backbone_size == "xclip":
                raise NotImplementedError
            elif t_backbone_size == "clip_iqa+":
                b = clip_vitL14(pretrained=True)
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)
        if divide_head:
            for key in backbone:
                pre_pool = False #if key == "technical" else True
                if key not in self.backbone_preserve_keys:
                    continue
                b = VQAHead(pre_pool=pre_pool, **vqa_head)
                print("Setting head:", key + "_head")
                setattr(self, key + "_head", b)
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)
        self.smtc_gate_tech = CrossGatingBlock(x_features=768, num_channels=768, block_size=1, 
                              grid_size=1, upsample_y=False, dropout_rate=0.1, use_bias=True, use_global_mlp=False)
        self.smtc_gate_aesc = CrossGatingBlock(x_features=768, num_channels=768, block_size=1, 
                              grid_size=1, upsample_y=False, dropout_rate=0.1, use_bias=True, use_global_mlp=False)

    def forward(
        self,
        vclips,
        inference=True,
        return_pooled_feats=False,
        return_raw_feats=False,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        assert (return_pooled_feats & return_raw_feats) == False, "Please only choose one kind of features to return"
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in vclips:
                    if key == 'technical' or key == 'aesthetic':
                        feat = getattr(self, key.split("_")[0] + "_backbone")(
                            vclips[key], multi=self.multi, layer=self.layer, **kwargs
                        )
                        if key == 'technical':
                            feat_gated = self.smtc_gate_tech(feats['semantic'], feat)
                        elif key == 'aesthetic':
                            feat_gated = self.smtc_gate_aesc(feats['semantic'], feat)
                        if hasattr(self, key.split("_")[0] + "_head"):
                            scores += [getattr(self, key.split("_")[0] + "_head")(feat_gated)]
                        else:
                            scores += [getattr(self, "vqa_head")(feat_gated)]
                    elif key == 'semantic':
                        x = vclips[key].squeeze()
                        x =  x.permute(1,0,2,3)
                        feat, _ = getattr(self, key.split("_")[0] + "_backbone")(
                            x, multi=self.multi, layer=self.layer, **kwargs
                        )
                        # for image feature from clipiqa+ VIT14
                        # image feature shape (t, c) -> (16, 768)
                        feat = feat.permute(1,0).contiguous() # (c, t) -> (768, 16)
                        feat = feat.unsqueeze(-1).unsqueeze(-1) # (c, t, w, h) -> (768, 16, 1, 1)
                        feat_expand = feat.expand(-1, -1, 7, 7) # (c, t, w, h) -> (768, 16, 7, 7)
                        feat_expand = feat_expand.unsqueeze(0) # (b, c, t, w, h) -> (1, 768, 16, 7, 7)
                        if hasattr(self, key.split("_")[0] + "_head"):
                            score = getattr(self, key.split("_")[0] + "_head")(feat_expand)
                        else:
                            score = getattr(self, "vqa_head")(feat_expand)
                        scores += [score]
                        feats[key] = feat_expand
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            if return_pooled_feats or return_raw_feats:
                return scores, feats
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                if key == 'technical' or key == 'aesthetic':
                    feat = getattr(self, key.split("_")[0] + "_backbone")(
                        vclips[key], multi=self.multi, layer=self.layer, **kwargs
                    )
                    if key == 'technical':
                        feat_gated = self.smtc_gate_tech(feats['semantic'], feat)
                    elif key == 'aesthetic':
                        feat_gated = self.smtc_gate_aesc(feats['semantic'], feat)
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat_gated)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat_gated)]
                    feats[key] = feat
                elif key == 'semantic':
                    scores_semantic_list = []
                    feats_semantic_list = []
                    for batch_idx in range(vclips[key].shape[0]):
                        x = vclips[key][batch_idx].squeeze()
                        x =  x.permute(1,0,2,3)
                        feat, _ = getattr(self, key.split("_")[0] + "_backbone")(
                            x, multi=self.multi, layer=self.layer, **kwargs
                        )
                        # for image feature from clipiqa+ VIT14
                        # image feature shape (t, c) -> (16, 768)
                        feat = feat.permute(1,0).contiguous() # (c, t) -> (768, 16)
                        feat = feat.unsqueeze(-1).unsqueeze(-1) # (c, t, w, h) -> (768, 16, 1, 1)
                        feat_expand = feat.expand(-1, -1, 7, 7) # (c, t, w, h) -> (768, 16, 7, 7)
                        feats_semantic_list.append(feat_expand)
                        if hasattr(self, key.split("_")[0] + "_head"):
                            feat_expand = feat_expand.unsqueeze(0) # (b, c, t, w, h) -> (1, 768, 16, 7, 7)
                            score = getattr(self, key.split("_")[0] + "_head")(feat_expand)
                            score = score.squeeze(0)
                            scores_semantic_list.append(score)
                        else:
                            feat_expand = feat_expand.unsqueeze(0) # (b, c, t, w, h) -> (1, 768, 16, 7, 7)
                            score = getattr(self, "vqa_head")(feat_expand)
                            score = score.squeeze(0)
                            scores_semantic_list.append(score)
                    scores_semantic_tensor = torch.stack(scores_semantic_list)
                    feats[key] = torch.stack(feats_semantic_list)
                    scores += [scores_semantic_tensor]
                if return_pooled_feats:
                    feats[key] = feat.mean((-3, -2, -1))
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores

    def forward_head(
        self,
        feats,
        inference=True,
        reduce_scores=False,
        pooled=False,
        **kwargs
    ):
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []
                feats = {}
                for key in feats:
                    feat = feats[key]
                    if hasattr(self, key.split("_")[0] + "_head"):
                        scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1, 2, 3, 4))
            self.train()
            return scores
        else:
            self.train()
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0] + "_backbone")(
                    vclips[key], multi=self.multi, layer=self.layer, **kwargs
                )
                if hasattr(self, key.split("_")[0] + "_head"):
                    scores += [getattr(self, key.split("_")[0] + "_head")(feat)]
                else:
                    scores += [getattr(self, "vqa_head")(feat)]
                if return_pooled_feats:
                    feats[key] = feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1, 2, 3, 4))
                    print(scores.shape)

            if return_pooled_feats:
                return scores, feats
            return scores
        
class MinimumCOVER(nn.Module):
    def __init__(self):
        super().__init__()
        self.technical_backbone = VideoBackbone()
        self.aesthetic_backbone = convnext_3d_tiny(pretrained=True)
        self.technical_head = VQAHead(pre_pool=False, in_channels=768)
        self.aesthetic_head = VQAHead(pre_pool=False, in_channels=768)


    def forward(self,aesthetic_view, technical_view):
        self.eval()
        with torch.no_grad():
            aesthetic_score = self.aesthetic_head(self.aesthetic_backbone(aesthetic_view))
            technical_score = self.technical_head(self.technical_backbone(technical_view))
            
        aesthetic_score_pooled = torch.mean(aesthetic_score, (1,2,3,4))
        technical_score_pooled = torch.mean(technical_score, (1,2,3,4))
        return [aesthetic_score_pooled, technical_score_pooled]



class BaseImageEvaluator(nn.Module):
    def __init__(
        self, backbone=dict(), iqa_head=dict(),
    ):
        super().__init__()
        self.backbone = ImageBackbone(**backbone)
        self.iqa_head = IQAHead(**iqa_head)

    def forward(self, image, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(image)
                score = self.iqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(image)
            score = self.iqa_head(feat)
            return score

    def forward_with_attention(self, image):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(image, require_attn=True)
            score = self.iqa_head(feat)
            return score, avg_attns

class CrossGatingBlock(nn.Module):  #input shape: n, c, h, w
    """Cross-gating MLP block."""
    def __init__(self, x_features, num_channels, block_size, grid_size, cin_y=0,upsample_y=True, use_bias=True, use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate
        self.Conv_0 = nn.Linear(self.x_features, self.num_channels)
        self.Conv_1 = nn.Linear(self.num_channels, self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu1 = nn.GELU(approximate='tanh')
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)
    def forward(self, x,y):     #n,c,t,h,w
        # Upscale Y signal, y is the gating signal.
        assert y.shape == x.shape
        x = x.permute(0,2,3,4,1).contiguous()  #n,t,h,w,c
        y = y.permute(0,2,3,4,1).contiguous()  #n,t,h,w,c
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        shortcut_y = y
        x = self.in_project_x(x)
        gx = self.gelu1(x)
        # Apply cross gating
        y = y * gx  # gating y using x
        y = self.out_project_y(y)
        y = self.dropout1(y)
        y = y + shortcut_y # y = y * x + y
        return y.permute(0,4,1,2,3).contiguous()  #n,c,t,h,w