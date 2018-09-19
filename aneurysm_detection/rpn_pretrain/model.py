import tensorflow as tf
import numpy as np
import aneurysm_detection.rpn_pretrain.utils as utils
import aneurysm_detection.rpn_pretrain.config as cfg

class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3], name='X')
        self.Y = tf.placeholder(tf.float32, [5,], name='Y')


        ##############################################
        self.logit, self.mean, self.gamma = self.model()
        # tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        # self.reconstruction_loss = tf.losses.mean_squared_error(utils.flatten('X_flatten', self.X), self.logit)
        self.reconstruction_loss = tf.reduce_sum(tf.squared_difference(utils.flatten('logit_flatten', tf.sigmoid(self.logit)),
                                                                       utils.flatten('X_flatten', tf.sigmoid(self.X))),
                                                 1)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.gamma) + tf.square(self.mean) - 1 - self.gamma, 1)
        # self.reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits
        # self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
        #############################################

    def model(self):

        inputs = self.X
        label = self.Y

        channel_n = cfg.INIT_N_FILTER
        inputs = self.feature_extractor(inputs, channel_n, cfg.PRETRAIN_N_LAYERS, cfg.N_DOWNSAMPLING)

        return inputs


    def feature_extractor(self, inputs, channel_n, n_layer, n_downsampling):
        with tf.variable_scope('feature_extractor_pretrain'):
            l = inputs
            for idx in range(n_layer):
                l = utils.residual_block_dw_dr(name='downconv_{}'.format(idx),
                                               inputs=l,
                                               channel_n=channel_n,
                                               width_mul=1.0,
                                               group_n=cfg.GROUP_N,
                                               drop_rate=self.drop_rate,
                                               act_fn=cfg.ACTIVATION_FUNC,
                                               norm_type=cfg.NORMALIZATION_TYPE,
                                               training=self.training,
                                               idx=idx)
                if idx + 1 <= n_downsampling:
                    l = utils.maxpool(name='maxpool_{}'.format(idx),
                                      inputs=l,
                                      pool_size=[2, 2],
                                      strides=[2, 2],
                                      padding='same')
                channel_n *= 2
                print(l)
        return l

    def rpn_bbox_generator(self, rpn_feature_maps, channel_n, anchors_per_location, drop_rate, training):
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
                else:
                    rpn_refinements = tf.concat((rpn_refinements, rpn_refinement), axis=1)
                    rpn_probs = tf.concat((rpn_probs, rpn_prob), axis=1)
                    rpn_class_logitss = tf.concat((rpn_class_logitss, rpn_class_logits), axis=1)

        return rpn_class_logitss, rpn_probs, rpn_refinements

    def region_proposal_network(self, anchors, rpn_refinements, rpn_probs, is_train):
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
            boxes = utils.apply_box_deltas_graph(pre_nms_anchors, deltas_slice)

            # clip box to range 0..1 [batch, N, (y1, x1, y2, x2)]
            window = np.array([0, 0, 1, 1], dtype=np.float32)
            boxes = utils.clip_boxes_graph(boxes, window)

            proposal = nms(boxes, scores_slice)
            proposal = tf.expand_dims(proposal, 0)

            if i == 0:
                proposals = proposal
            else:
                proposals = tf.concat([proposals, proposal], 0)

        return proposals

    def build_rpn_targets(self, anchors, gt_boxes, config):
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
        overlaps = utils.compute_overlaps(anchors, gt_boxes)

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

