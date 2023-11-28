import numpy as np
import math
import torch
import torch.nn as nn
from .encoder import build_encoder
from .decoder import build_decoder

from ..backbones import backbone
from .deformable_transformer import build_deforamble_transformer
from .transformer import build_transformer

from utils.nms import multiclass_nms
from util.misc import NestedTensor, nested_tensor_from_tensor_list
import torch.nn.functional as F
from utils.misc import load_init

DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class SNPMiSo(nn.Module):
    trainable: bool

    def __init__(self, 
                 cfg,
                 args,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(SNPMiSo, self).__init__()
        self.cfg = cfg
        self.device = device
        self.fmp_size = None
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])
        self.num_anchors = len(cfg['anchor_size'])

        hidden_dim1 = build_deforamble_transformer(args).d_model  # 256
        hidden_dim2 = build_transformer(args).d_model         # 512

        self.input_proj1 = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(backbone.build_backbone(cfg=cfg, args=args).num_channels[-2],
                      hidden_dim1, kernel_size=1),
            nn.GroupNorm(32, hidden_dim1)
            )
        self.input_proj2 = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(backbone.build_backbone(cfg=cfg, args=args).num_channels[-1],
                      hidden_dim2, kernel_size=1)
            )

        for proj in [self.input_proj1, self.input_proj2]:
            nn.init.xavier_uniform_(proj[1].weight, gain=1)
            nn.init.constant_(proj[1].bias, 0)                 # 初始化从主干网络到编码器中间的卷积层

        self.output_up = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512)
        )

        nn.init.normal_(self.output_up[1].weight, std=0.02)   # 初始化上采样层
        nn.init.constant_(self.output_up[1].bias, 0)

        self.dim_change1 = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(backbone.build_backbone(cfg=cfg, args=args).num_channels[-2],
                      hidden_dim1, kernel_size=1),
            nn.BatchNorm2d(hidden_dim1)                           # 改变通道数
        )
        self.dim_change2 = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(backbone.build_backbone(cfg=cfg, args=args).num_channels[-1],
                      hidden_dim2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim2)                           # 改变通道数
        )

        for chan in [self.dim_change1, self.dim_change2]:
            nn.init.xavier_uniform_(chan[1].weight, gain=1)
            nn.init.constant_(chan[1].bias, 0)                    # 初始化


        #-------------------------- Network -----------------------------#
        ## backbone
        self.backbone = backbone.build_backbone(cfg=cfg, args=args)

        ## neck
        self.transformer1 = build_deforamble_transformer(args)
        self.transformer2 = build_transformer(args)
        self.neck = build_encoder(cfg=cfg, in_dim=hidden_dim1 + hidden_dim2, out_dim=cfg['encoder_dim'], st=2,
                                  dl=cfg['dilation_list'])
                                     
        ## head
        self.head = build_decoder(cfg, cfg['decoder_dim'], num_classes, self.num_anchors)


    def generate_anchors(self, fmp_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # check anchor boxes
        if self.fmp_size is not None and self.fmp_size == fmp_size:
            return self.anchor_boxes
        else:
            # generate grid cells
            fmp_h, fmp_w = fmp_size
            anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
            # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
            anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
            anchor_xy *= self.stride

            # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
            anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

            # [HW, KA, 4] -> [M, 4]
            anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
            anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

            self.anchor_boxes = anchor_boxes
            self.fmp_size = fmp_size

            return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4]
            pred_reg: (List[tensor]) [B, M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                      max=self.cfg['ctr_clamp'],
                                      min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, 
                                max=DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process(self, cls_pred, reg_pred, anchors):
        """
        Input:
            cls_pred: (Tensor) [H x W x KA, C]
            reg_pred: (Tensor) [H x W x KA, 4]
            anchors:  (Tensor) [H x W x KA, 4]
        """
        # (HxWxAxK,)
        cls_pred = cls_pred.flatten().sigmoid_()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, reg_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = cls_pred.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        labels = topk_idxs % self.num_classes

        reg_pred = reg_pred[anchor_idxs]
        anchors = anchors[anchor_idxs]

        # decode bbox
        bboxes = self.decode_boxes(anchors, reg_pred)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

        return bboxes, scores, labels


    @torch.no_grad()
    def inference_single_image(self, x, bmask):
        img_h, img_w = x.shape[2:]
        if bmask is None:
            samples = nested_tensor_from_tensor_list(x)
        else:
            samples = NestedTensor(x, bmask)
        srcs = []
        masks = []
        poss = []

        # backbone
        features, pos1, pos2 = self.backbone(samples)
        src1, mask1 = features[-2].decompose()
        srcs.append(self.input_proj1(src1))
        masks.append(mask1)
        poss.append(pos1[-2])
        src2, mask2 = features[-1].decompose()
        assert mask1 is not None and mask2 is not None

        # neck
        memory1 = self.transformer1(srcs, masks, poss)
        m1_h, m1_w = memory1.shape[2], memory1.shape[3]
        memory1 = memory1 + self.dim_change1(src1)
        memory2 = self.transformer2(self.input_proj2(src2), mask2, pos2[-1])
        memory2 = memory2 + self.dim_change2(src2)
        memory2 = self.output_up(memory2)
        memory2 = F.interpolate(memory2, size=(m1_h, m1_w), mode='bilinear')
        memory_cat = torch.cat([memory1, memory2], dim=1)
        x = self.neck(memory_cat)
        fmp_h, fmp_w = x.shape[2:]

        # head
        cls_pred, reg_pred = self.head(x)
        cls_pred, reg_pred = cls_pred[0], reg_pred[0]

        # anchor box
        anchor_boxes = self.generate_anchors(fmp_size=[fmp_h, fmp_w]) # [M, 4]

        # post process
        bboxes, scores, labels = self.post_process(cls_pred, reg_pred, anchor_boxes)

        # normalize bbox
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels


    def forward(self, x, bmask=None, amask=None):
        if not self.trainable:
            return self.inference_single_image(x, bmask)
        else:
            samples = NestedTensor(x, bmask)
            srcs = []
            masks = []
            poss = []
            # backbone
            features, pos1, pos2 = self.backbone(samples)
            src1, mask1 = features[-2].decompose()
            srcs.append(self.input_proj1(src1))
            masks.append(mask1)
            poss.append(pos1[-2])
            src2, mask2 = features[-1].decompose()
            assert mask1 is not None and mask2 is not None

            # neck
            memory1 = self.transformer1(srcs, masks, poss)
            m1_h, m1_w = memory1.shape[2], memory1.shape[3]
            memory1 = memory1 + self.dim_change1(src1)
            memory2 = self.transformer2(self.input_proj2(src2), mask2, pos2[-1])
            memory2 = memory2 + self.dim_change2(src2)
            memory2 = self.output_up(memory2)
            memory2 = F.interpolate(memory2, size=(m1_h, m1_w), mode='bilinear')
            memory_cat = torch.cat([memory1, memory2], dim=1)
            x = self.neck(memory_cat)
            fmp_h, fmp_w = x.shape[2:]

            # head
            cls_pred, reg_pred = self.head(x)

            # anchor box: [M, 4]
            anchor_boxes = self.generate_anchors(fmp_size=[fmp_h, fmp_w])

            # decode box: [B, M, 4]
            box_pred = self.decode_boxes(anchor_boxes[None], reg_pred)
            
            if amask is not None:
                # [B, H, W]
                amask = torch.nn.functional.interpolate(amask[None], size=[fmp_h, fmp_w]).bool()[0]
                # [B, H, W] -> [B, HW]
                amask = amask.flatten(1)
                # [B, HW] -> [B, HW, KA] -> [BM,], M= HW x KA
                amask = amask[..., None].repeat(1, 1, self.num_anchors).flatten()

            outputs = {"pred_cls": cls_pred,
                       "pred_box": box_pred,
                       "anchors": anchor_boxes,
                       "mask": amask}

            return outputs 
