import tensorflow as tf
import utils as utils
import numpy as np
import aneurysm_detection.rpn_pretrain.config as cfg
import math

class Model_Resnet:
    def __init__(self, mode='training'):
        self.mode = mode
        self.training = True if self.mode == 'training' else False
        self.config = cfg

        with tf.name_scope('input'):
            self._set_placeholders()

        self.outputs = self.build()
        self.loss = self.outputs['rpn_class_loss'] + self.outputs['rpn_bbox_loss']\
                    # + self.outputs['class_loss'] \
                    # + self.outputs['bbox_loss'] + self.outputs['mask_loss']

        tf.logging.info('Setting up the main structure')

    def _set_placeholders(self):
        # input placeholder
        # None : Anchor BBox
        ANCHOR_BBOX = None
        NUM_INSTANCES = None

        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.input_image = tf.placeholder(dtype=tf.float32,
                                          shape=[cfg.BATCH_SIZE, cfg.IMAGE_MAX_DIM, cfg.IMAGE_MAX_DIM, cfg.IMAGE_CHANNEL])

        self.gt_boxes = tf.placeholder(dtype=tf.float32,
                                       shape=[cfg.BATCH_SIZE, NUM_INSTANCES, 4])
        self.gt_masks = tf.placeholder(dtype=tf.float32,
                                       shape=[cfg.BATCH_SIZE, cfg.IMAGE_MAX_DIM, cfg.IMAGE_MAX_DIM, NUM_INSTANCES])
        self.rpn_match = tf.placeholder(dtype=tf.float32, shape=[cfg.BATCH_SIZE, ANCHOR_BBOX])
        self.rpn_bbox = tf.placeholder(dtype=tf.float32, shape=[cfg.BATCH_SIZE, ANCHOR_BBOX, 4])
        # self.image_meta = tf.placeholder(dtype=tf.float32, shape=[cfg.BATCH_SIZE, cfg.IMAGE_META_SIZE])

    ###
    # ResNet 152-layer : 3-48-36-3
    # 101-layer : 3-4-23-3
    # 50-layer : 3-4-6-3
    ###
    def build(self):
        # Image size must be dividable by 2 multiple times
        h, w = cfg.IMG_SIZE

        # Inputs
        input_image = self.input_image
        input_rpn_bbox = self.rpn_bbox
        input_rpn_match = tf.expand_dims(self.rpn_match, axis=-1)




        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Create RPN BBox / Classifier
        # anchors = self.anchors
        anchors = get_anchors(cfg.IMG_SIZE, cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS, cfg.FEATURE_STRIDES, cfg.ANCHOR_STRIDE)


        # 각각 [P2 ~ P6]의 결과. (P2 > P3 > ..)
        rpn_class_logits, rpn_probs, rpn_refinements = generate_rpn_bbox(rpn_feature_maps,
                                                                         cfg.RPN_N_FILTER,
                                                                         len(cfg.ANCHOR_RATIOS),
                                                                         drop_rate=0,####################
                                                                         training=True) ########################

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposals = region_proposal_network(cfg, anchors, rpn_refinements, rpn_probs, self.training)

        ## Loss ##
        rpn_class_loss = rpn_class_loss_graph(input_rpn_match, rpn_class_logits)
        rpn_bbox_loss = rpn_bbox_loss_graph(cfg, input_rpn_bbox, input_rpn_match, rpn_refinements)
        # class_loss = mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits)
        # bbox_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, mrcnn_bbox, self.config)
        # mask_loss = mrcnn_mask_loss_graph(target_mask, target_class_ids, mrcnn_mask)

        output = {
            'proposals': proposals,
            'rpn_class_loss': rpn_class_loss,
            'rpn_bbox_loss': rpn_bbox_loss
        }

        return output

def get_anchors(image_shape, scales, ratios, feature_strides, anchor_stride):
    """Returns anchor pyramid for the given image size."""
    feature_shapes = compute_backbone_shapes(image_shape, feature_strides)
    a = generate_pyramid_anchors(
        scales,
        ratios,
        feature_shapes,
        feature_strides,
        anchor_stride)
    # Normalize coordinates
    return norm_boxes(a, image_shape)

def compute_backbone_shapes(image_shape, backbone_strides):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # backbone_stride = [4, 8, 16, 32, 64]
    return np.array([[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
                     for stride in backbone_strides])

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    # scales : (1,) -> (3,) [128, 128, 128]
    # ratios : (3,) -> (3,) [0.5, 1, 2]
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    # shape : feature_shape
    # feature_stride : feature_stride
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def generate_rpn_bbox(rpn_feature_maps, channel_n, anchors_per_location, drop_rate, training):
    '''

    :param rpn_feature_maps: [P2, P3, P4, P5, (P6)]
    :param anchor_stride: same as anchors
    :param anchors_per_location: anchor per pixel
    :return:
    rpn_class_logitss : [batch, total anchors_cnt, 2]
    rpn_probs : [batch, total anchors_cnt, 2]
                2 : probability of foreground/background
    rpn_refinements : [batch, total anchors_cnt, 4]
                4 : dy, dx, log(dh), log(dw) (refinement of each bbox)

    # rpn_class_logitss : [batch, each anchors_cnt, 2] * len(anchors_cnt)
    # rpn_probs : [batch, each anchors_cnt, 2] * len(anchors_cnt)
    #             2 : probability of foreground/background
    # rpn_refinements : [batch * length of rpn_feature_maps, h * w * anchors per location, 4]
    #             4 : dy, dx, log(dh), log(dw) (refinement of each bbox)

    '''
    with tf.variable_scope('rpn_pretrain'):

        rpn_class_logitss = ''
        rpn_probs = ''
        rpn_refinements = ''

        for idx, p in enumerate(rpn_feature_maps):

            shared = utils.residual_block_dw_dr(name='rpn_shared_{}'.format(idx),
                                                inputs=p,
                                                channel_n=channel_n,
                                                width_mul=1.0,
                                                group_n=cfg.GROUP_N,
                                                drop_rate=drop_rate,
                                                act_fn=cfg.ACTIVATION_FUNC,
                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                training=training,
                                                idx=idx)
            l = utils.residual_block_dw_dr(name='rpn_class_{}'.format(idx),
                                           inputs=shared,
                                           channel_n=anchors_per_location * 2,
                                           width_mul=1.0,
                                           group_n=cfg.GROUP_N,
                                           drop_rate=drop_rate,
                                           act_fn=cfg.ACTIVATION_FUNC,
                                           norm_type=cfg.NORMALIZATION_TYPE,
                                           training=training,
                                           idx=idx)

            rpn_class_logits = tf.reshape(l, (tf.shape(l)[0], -1, 2))
            rpn_prob = tf.nn.softmax(rpn_class_logits)

            l = utils.residual_block_dw_dr(name='rpn_bbox_{}'.format(idx),
                                           inputs=shared,
                                           channel_n=anchors_per_location * 4,
                                           width_mul=1.0,
                                           group_n=cfg.GROUP_N,
                                           drop_rate=drop_rate,
                                           act_fn=cfg.ACTIVATION_FUNC,
                                           norm_type=cfg.NORMALIZATION_TYPE,
                                           training=training,
                                           idx=idx)

            rpn_refinement = tf.reshape(l, (tf.shape(l)[0], -1, 4))

            if idx == 0:
                rpn_refinements = rpn_refinement
                rpn_probs = rpn_prob
                rpn_class_logitss = rpn_class_logits
            else :
                rpn_refinements = tf.concat((rpn_refinements, rpn_refinement), axis=1)
                rpn_probs = tf.concat((rpn_probs, rpn_prob), axis=1)
                rpn_class_logitss = tf.concat((rpn_class_logitss, rpn_class_logits), axis=1)

    return rpn_class_logitss, rpn_probs, rpn_refinements



def region_proposal_network(config, anchors, rpn_refinements, rpn_probs, is_train):
    '''

    - label data -
    :param anchors: [total anchors_cnt, 4] -> y1, x1, y2, x2
    :param anchors_refinements: [batch, total anchors_cnt, 4] -> x, y, log(dw), log(dh) (예정)
    :param anchor_has_object: [batch, total anchors_cnt]

    - output of model -
    :param rpn_refinements: [batch, total anchors_cnt, 2] -> x, y, log(dw), log(dh)
    :param rpn_probs: [batch, total anchors_cnt, 2] -> (back, fore)

    :return: [batch, self.proposal_count, 4]
    '''
    '''
    1. get foreground's score(prob[:,:,0]) / delta(refinements)
    2. 상위 6000개의 anchor만 남기기(use tf.nn.top_k, indices, tf.gather)
    3. applying refinements on box (refinements : dy, dx, log(dh), log(dw))
    4. clip image boundary (in normalized coordinates(0.0.1.1)?)
    5. NMS(Non-max suppression) (use tf.image.non_max_suppression) and use pad if has not enough proposals.
    6. output shape : (bs, self.proposal_count, 4)
    '''
    proposals = ''
    proposal_count = cfg.POST_NMS_ROIS_TRAINING if is_train \
        else cfg.POST_NMS_ROIS_INFERENCE
    nms_threshold = cfg.RPN_NMS_THRESHOLD

    def nms(boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, proposal_count,
            nms_threshold, name="rpn_non_max_suppression")
        proposals = tf.gather(boxes, indices)
        # Pad if needed
        padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals

    deltas = rpn_refinements
    scores = rpn_probs[:, :, 1]

    for i in range(cfg.BATCH_SIZE):
        # deltas_slice: [total anchors_cnt, 4] -> x, y, log(dw), log(dh)
        # scores_slice: [total anchors_cnt]
        deltas_slice = deltas[i]
        scores_slice = scores[i]

        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores_slice, pre_nms_limit, sorted=True, name='top_anchors').indices
        scores_slice = tf.gather(scores_slice, ix)
        deltas_slice = tf.gather(deltas_slice, ix)
        pre_nms_anchors = tf.gather(anchors, ix)

        # apply deltas to anchors to get refined anchors.
        boxes = apply_box_deltas_graph(pre_nms_anchors, deltas_slice)

        # clip box to range 0..1 [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1 ,1], dtype=np.float32)
        boxes = clip_boxes_graph(boxes, window)

        proposal = nms(boxes, scores_slice)
        proposal = tf.expand_dims(proposal, 0)

        if i == 0:
            proposals = proposal
        else:
            proposals = tf.concat([proposals, proposal], 0)

    return proposals

def build_rpn_targets(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    오버랩 계산/positive anchors 계산
    rpn_match : anchor들과 gt_box를 비교하여 positive/negative 판별
    rpn_bbox : delta : anchor box를 gt_box에 맞추기 위한 보정치(delta) 계산

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]
        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes (넓이)
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped