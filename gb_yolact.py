import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, conv1x1, conv3x3
import numpy as np
from functools import partial
from itertools import product, chain
from math import sqrt
from typing import List, Tuple

use_torch2trt = False

use_jit = False if use_torch2trt else torch.cuda.device_count() <= 1
NoneTensor = None if use_torch2trt else torch.Tensor()

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn

class InterpolateModule(nn.Module):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		return F.interpolate(x, *self.args, **self.kwdargs)

def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels
        
        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels

class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

        return layer

    def forward(self, x, partial:bool=False):
        """ Returns a list of convouts for each layer. """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        layer_idx = 0
        for layer in self.layers:
            layer_idx += 1
            if not partial or layer_idx <= 2:
                x = layer(x)
                outs.append(x)
        return outs

    def init_backbone(self, path, map_location=None):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path, map_location=map_location)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx-1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)

class FPN_phase_1(ScriptModuleWrapper):
    __constants__ = ['interpolation_mode', 'lat_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.src_channels = in_channels

        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, 256, kernel_size=1)
            for x in reversed(in_channels)
        ])

        self.interpolation_mode = 'bilinear'

    @script_method_wrapper
    def forward(self, x1=NoneTensor, x2=NoneTensor, x3=NoneTensor, x4=NoneTensor, x5=NoneTensor, x6=NoneTensor, x7=NoneTensor):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        convouts_ = [x1, x2, x3, x4, x5, x6, x7]
        convouts = []
        j = 0
        while j < len(convouts_):
            if convouts_[j] is not None and convouts_[j].size(0):
                convouts.append(convouts_[j])
            j += 1
        # convouts = [x for x in convouts if x is not None]

        out = []
        lat_layers = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)
            lat_layers.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            lat_j = lat_layer(convouts[j])
            lat_layers[j] = lat_j
            x = x + lat_j
            out[j] = x
        
        for i in range(len(convouts)):
            out.append(lat_layers[i])
        return out

class FPN_phase_2(ScriptModuleWrapper):
    __constants__ = ['num_downsample', 'use_conv_downsample', 'pred_layers', 'downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.src_channels = in_channels

        # This is here for backwards compatability
        padding = 1
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
            for _ in range(2)
        ])

        self.num_downsample = 2
        self.use_conv_downsample = True

    @script_method_wrapper
    def forward(self, x1=NoneTensor, x2=NoneTensor, x3=NoneTensor, x4=NoneTensor, x5=NoneTensor, x6=NoneTensor, x7=NoneTensor):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        # out = [x1, x2, x3, x4, x5, x6, x7]
        # out = [x for x in out if x is not None]

        out_ = [x1, x2, x3, x4, x5, x6, x7]
        out = []
        j = 0
        while j < len(out_):
            if out_[j] is not None and out_[j].size(0):
                out.append(out_[j])
            j += 1

        len_convouts = len(out)

        j = len_convouts
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out

class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample',
                     'lat_layers', 'pred_layers', 'downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
                for _ in range(2)
            ])
        
        self.interpolation_mode  = 'bilinear'
        self.num_downsample      = 2
        self.use_conv_downsample = True

    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            x = x + lat_layer(convouts[j])
            out[j] = x

        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        # In the original paper, this takes care of P6
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        return out

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0, mask_dim=None, num_classes=81):
        super().__init__()

        self.params = [in_channels, out_channels, aspect_ratios, scales, parent, index]

        self.num_classes = num_classes
        self.mask_dim    = mask_dim
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index

        self.extra_head_net = [(256, 3, {'padding': 1})]
        self.head_layer_params = {'kernel_size': 3, 'padding': 1}
        self.num_instance_coeffs = 64

        if parent is None:
            if self.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, self.extra_head_net)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **(self.head_layer_params))
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **(self.head_layer_params))
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **(self.head_layer_params))
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))
            extra_layers = (0, 0, 0)
            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in extra_layers]

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if self.extra_head_net is not None:
            x = src.upfeature(x)

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)

        mask = torch.tanh(mask)
        
        priors = self.make_priors(conv_h, conv_w)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
        
        return preds

class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions, extras=None):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data, extras)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]
                
                out.append(result)
        
        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data, extras=None):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
    
        if scores.size(1) == 0:
            return None
        
        if self.use_fast_nms:
            boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}
    

    def coefficient_nms(self, coeffs, scores, cos_threshold=0.9, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        coeffs_norm = F.normalize(coeffs[idx], dim=1)

        # Compute the pairwise cosine similarity between the coefficients
        cos_similarity = coeffs_norm @ coeffs_norm.t()
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        cos_similarity.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the cos similarity matrix along the columns, each column will represent the
        # maximum cosine similarity between this element and every element with a higher
        # score than this element.
        cos_max, _ = torch.max(cos_similarity, dim=0)

        # Now just filter out the ones higher than the threshold
        idx_out = idx[cos_max <= cos_threshold]


        # new_mask_norm = F.normalize(masks[idx_out], dim=1)
        # print(new_mask_norm[:5] @ new_mask_norm[:5].t())
        
        return idx_out, idx_out.size(0)

    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)

        # This try-except block aims to fix the IndexError that we might encounter when we train on custom datasets and evaluate with TensorRT enabled. See https://github.com/haotian-liu/yolact_edge/issues/27.
        try:
            classes = classes[keep]
        except IndexError:

            import logging
            logger = logging.getLogger("yolact.layers.detect")
            logger.warning("Encountered IndexError as mentioned in https://github.com/haotian-liu/yolact_edge/issues/27. Flattening predictions to avoid error, please verify the outputs. If there are any problems you met related to this, please report an issue.")

            classes = torch.flatten(classes, end_dim=1)
            boxes = torch.flatten(boxes, end_dim=1)
            masks = torch.flatten(masks, end_dim=1)
            scores = torch.flatten(scores, end_dim=1)
            keep = torch.flatten(keep, end_dim=1)

            classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        try:
            classes = classes[idx]
            boxes = boxes[idx]
            masks = masks[idx]
        except IndexError:

            import logging
            logger = logging.getLogger("yolact.layers.detect")
            logger.warning("Encountered IndexError as mentioned in https://github.com/haotian-liu/yolact_edge/issues/27. Using `torch.index_select` to avoid error, please verify the outputs. If there are any problems you met related to this, please report an issue.")

            classes = torch.index_select(classes, 0, idx)
            boxes = torch.index_select(boxes, 0, idx)
            masks = torch.index_select(masks, 0, idx)

        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores


class yolactedge(nn.Module):
    def __init__(self, training=False):
        super().__init__()

        self.training = training

        self.fpn_num_features = 256

        self.backbone = ResNetBackbone([3, 4, 23, 3])
        num_layers = max(list(range(1, 4))) + 1
        while len(self.backbone.layers) < num_layers:
            self.backbone.add_layer()

        self.num_grids = 0

        self.proto_src = 0

        self.num_classes = 81
        
        in_channels = self.fpn_num_features
        in_channels += self.num_grids
        
        # The include_last_relu=false here is because we might want to change it to another function
        mask_proto_net = [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
        self.proto_net, self.mask_dim = make_net(in_channels, mask_proto_net, include_last_relu=False)

        self.selected_layers = list(range(1, 4))
        src_channels = self.backbone.channels
        
        self.fpn_phase_1 = FPN_phase_1([src_channels[i] for i in self.selected_layers])
        self.fpn_phase_2 = FPN_phase_2([src_channels[i] for i in self.selected_layers])

        self.selected_layers = list(range(len(self.selected_layers) + 2))

        src_channels = [self.fpn_num_features] * len(self.selected_layers)

        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if idx > 0:
                parent = self.prediction_layers[0]

            pred_aspect_ratios = [ [[1, 1/2, 2]] ]*5
            pred_scales = [[24], [48], [96], [192], [384]]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = pred_aspect_ratios[idx],
                                    scales        = pred_scales[idx],
                                    parent        = parent,
                                    index         = idx,
                                    mask_dim      = self.mask_dim,
                                    num_classes   = self.num_classes)
            self.prediction_layers.append(pred)

        self.semantic_seg_conv = nn.Conv2d(src_channels[0], self.num_classes-1, kernel_size=1)

        # # For use in evaluation
        # self.detect = Detect(self.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)

    def forward(self, inp):
        return inp

if __name__ == '__main__':
    print('hello world')
    net = yolactedge().cuda()


    dummy_input = torch.randn(10, 3, 550, 550, device='cuda')
    torch.onnx.export(net, dummy_input, "test.onnx", verbose=True)
    
