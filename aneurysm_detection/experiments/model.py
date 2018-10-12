import tensorflow as tf
import numpy as np
import utils as utils
import config as cfg

class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3], name='X')
        # self.Y = tf.placeholder(tf.float32, [5,], name='Y')
        self.anchors = tf.placeholder(tf.float32, [None, 4], name='anchors')
        self.rpn_class_label = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, None, 1], name='rpn_class_label')
        self.rpn_bbox_label = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, None, 4], name='rpn_bbox_label')
        self.detector_class_label = tf.placeholder(tf.float32, [None], name='detector_class_label')
        self.detector_bbox_label = tf.placeholder(tf.float32, [None, 4], name='detector_bbox_label')

        ##############################################

        self.rpn_class_logitss, self.rpn_bbox_refinements, \
        self.detector_class_logits, self.detector_bbox_refinements,\
            self.detector_class_label2, self.detector_bbox_label2 = self.model()
        self.rpn_class_loss = utils.rpn_class_loss_graph(self.rpn_class_label, self.rpn_class_logitss)
        self.rpn_bbox_loss = utils.rpn_bbox_loss_graph(cfg, self.rpn_bbox_label, self.rpn_class_label, self.rpn_bbox_refinements)

        self.detector_class_loss = utils.detector_class_loss_graph(self.detector_class_label2, self.detector_class_logits)
        self.detector_bbox_loss = utils.detector_bbox_loss_graph(self.detector_bbox_label2, self.detector_class_label2, self.detector_bbox_refinements, cfg)


        ############## lambda 값은 변경해야. 일단 0.25로 ##############
        self.loss = 0.25 * self.rpn_class_loss + 0.25 * self.rpn_bbox_loss + 0.25 * self.detector_class_loss + 0.25 * self.detector_bbox_loss
        # self.reconstruction_loss = tf.reduce_sum(tf.squared_difference(utils.flatten('logit_flatten', tf.sigmoid(self.logit)),
        #                                                                utils.flatten('X_flatten', tf.sigmoid(self.X))),
        #                                          1)
        # self.latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.gamma) + tf.square(self.mean) - 1 - self.gamma, 1)
        # self.loss = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
        #############################################

    def model(self):

        channel_n = cfg.INIT_N_FILTER

        ### Feature Extractor (Conv1~5) ###
        feature_maps = self.feature_extractor(self.X, channel_n, cfg.PRETRAIN_N_LAYERS)

        ### RPN ###
        rpn_feature_maps = tf.expand_dims(feature_maps[tf.shape(feature_maps)[0]//2], axis=0)
        rpn_feature_maps = [rpn_feature_maps]
        rpn_class_logitss, rpn_class_probs, rpn_bbox_refinements = self.rpn_bbox_generator(rpn_feature_maps,
                                                                                           cfg.RPN_N_FILTER,
                                                                                           len(cfg.ANCHOR_RATIOS))
        ### Additional Conv for 3d contexts ###
        with tf.variable_scope('common_conv_pretrain'):
            feature_maps = utils.residual_block_dw_dr(name='common_conv',
                                                      inputs=feature_maps,
                                                      channel_n=cfg.POOLED_SIZE[0]*cfg.POOLED_SIZE[1]*10,
                                                      width_mul=1.0,
                                                      group_n=cfg.GROUP_N,
                                                      drop_rate=self.drop_rate,
                                                      act_fn=cfg.ACTIVATION_FUNC,
                                                      norm_type=cfg.NORMALIZATION_TYPE,
                                                      training=self.training,
                                                      idx=0)
        print('feature_maps', feature_maps)
        feature_shape_h, feature_shape_w, feature_shape_c = feature_maps.get_shape().as_list()[1:]
        feature_maps = tf.expand_dims(feature_maps, axis=0)
        feature_maps = tf.transpose(feature_maps, (0, 2, 3, 1, 4))
        feature_maps = tf.reshape(feature_maps, (1, feature_shape_h, feature_shape_w, cfg.N_3D_CONTEXT // 3 * feature_shape_c))
        print('concated_feature_maps', feature_maps)


        ### Proposal Network ###
        proposals = self.region_proposal_network(self.anchors,
                                                 rpn_bbox_refinements,
                                                 rpn_class_probs,
                                                 self.training)
        # proposals = tf.squeeze(proposals, axis=0)

        ### Make detector label ###
        proposals, detector_class_label, detector_bbox_label = self.detection_targets_graph(proposals,
                                                                                             self.detector_class_label,
                                                                                             self.detector_bbox_label,
                                                                                             cfg)
        detector_class_label = tf.expand_dims(detector_class_label, axis=0)
        detector_bbox_label = tf.expand_dims(detector_bbox_label, axis=0)
        proposals = tf.expand_dims(proposals, axis=0)
        print('proposals', proposals)

        self.proposals = tf.identity(proposals)

        ### Detector ###
        detector_class_logits, detector_bbox_refinements = self.detector(proposals, feature_maps, feature_shape_c, cfg)
        print('detector_class_logits', detector_class_logits)
        # detector_class_logits = tf.reshape(detector_class_logits, (-1, config.TRAIN_ROIS_PER_IMAGE, config.N_CLASS))
        # detector_bbox_refinements = tf.reshape(detector_class_logits, (-1, config.TRAIN_ROIS_PER_IMAGE, config.N_CLASS, 4))
        print('detector_bbox_refinements', detector_bbox_refinements)


        ### Detector Final Outputs ###
        final_proposals = tf.squeeze(proposals, axis=0)
        final_detector_class_probs = tf.squeeze(tf.nn.softmax(detector_class_logits), axis=0)
        final_detector_bbox_refinements = tf.squeeze(detector_bbox_refinements, axis=0)

        self.detection_outputs = self.refine_detections_graph(final_proposals,
                                                              final_detector_class_probs,
                                                              final_detector_bbox_refinements,
                                                              cfg)

        return rpn_class_logitss, rpn_bbox_refinements, detector_class_logits, detector_bbox_refinements, detector_class_label, detector_bbox_label


    def feature_extractor(self, inputs, channel_n, n_layer):
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

                if channel_n < 2 ** 11:
                    channel_n *= 2

                if idx + 1 <= cfg.N_DOWNSAMPLING:
                    l = utils.maxpool(name='maxpool_{}'.format(idx),
                                      inputs=l,
                                      pool_size=[2, 2],
                                      strides=[2, 2],
                                      padding='same')
                print(l)
        return l

    def rpn_bbox_generator(self, rpn_feature_maps, channel_n, anchors_per_location):
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

                shared = utils.residual_block_dw_dr(name='rpn_shared_{}_0'.format(idx),
                                                    inputs=p,
                                                    channel_n=channel_n,
                                                    width_mul=1.0,
                                                    group_n=cfg.GROUP_N,
                                                    drop_rate=self.drop_rate,
                                                    act_fn=cfg.ACTIVATION_FUNC,
                                                    norm_type=cfg.NORMALIZATION_TYPE,
                                                    training=self.training,
                                                    idx=idx)
                print(shared)
                shared = utils.residual_block_dw_dr(name='rpn_shared_{}_1'.format(idx),
                                                    inputs=shared,
                                                    channel_n=channel_n,
                                                    width_mul=1.0,
                                                    group_n=cfg.GROUP_N,
                                                    drop_rate=self.drop_rate,
                                                    act_fn=cfg.ACTIVATION_FUNC,
                                                    norm_type=cfg.NORMALIZATION_TYPE,
                                                    training=self.training,
                                                    idx=idx)
                print(shared)
                shared = utils.residual_block_dw_dr(name='rpn_shared_{}_2'.format(idx),
                                                    inputs=shared,
                                                    channel_n=channel_n,
                                                    width_mul=1.0,
                                                    group_n=cfg.GROUP_N,
                                                    drop_rate=self.drop_rate,
                                                    act_fn=cfg.ACTIVATION_FUNC,
                                                    norm_type=cfg.NORMALIZATION_TYPE,
                                                    training=self.training,
                                                    idx=idx)
                print(shared)
                l = utils.depthwise_separable_convlayer_dr(name='rpn_class_{}'.format(idx),
                                                           inputs=shared,
                                                           channel_n=anchors_per_location * 2,
                                                           width_mul=1.0,
                                                           group_n=cfg.GROUP_N,
                                                           drop_rate=self.drop_rate,
                                                           act_fn=cfg.ACTIVATION_FUNC,
                                                           norm_type=cfg.NORMALIZATION_TYPE,
                                                           training=self.training,
                                                           idx=idx)

                print('rpn_class',l)
                rpn_class_logits = tf.reshape(l, (tf.shape(l)[0], -1, 2))
                rpn_prob = tf.nn.softmax(rpn_class_logits)

                l = utils.depthwise_separable_convlayer_dr(name='rpn_bbox_{}'.format(idx),
                                                           inputs=shared,
                                                           channel_n=anchors_per_location * 4,
                                                           width_mul=1.0,
                                                           group_n=cfg.GROUP_N,
                                                           drop_rate=self.drop_rate,
                                                           act_fn=cfg.ACTIVATION_FUNC,
                                                           norm_type=cfg.NORMALIZATION_TYPE,
                                                           training=self.training,
                                                           idx=idx)
                print('rpn_bbox',l)
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
        proposal_count = cfg.POST_NMS_ROIS_TRAINING if is_train == tf.constant(True) else cfg.POST_NMS_ROIS_INFERENCE
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
            boxes = utils.apply_box_deltas_graph2(pre_nms_anchors, deltas_slice)

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

    def detector(self, proposals, feature_maps, channel_n, config):
        with tf.variable_scope('detector_pretrain'):
            pooled_feature_maps = utils.roi_pooling(proposals, feature_maps, config.POOLED_SIZE, feature_pyramid=False)
            print('pooled', pooled_feature_maps)

            for i in range(3):
                pooled_feature_maps = utils.residual_block_dw_dr(name='detector_conv{}'.format(i),
                                                                 inputs=pooled_feature_maps,
                                                                 channel_n=channel_n,
                                                                 width_mul=1.0,
                                                                 group_n=cfg.GROUP_N,
                                                                 drop_rate=self.drop_rate,
                                                                 act_fn=cfg.ACTIVATION_FUNC,
                                                                 norm_type=cfg.NORMALIZATION_TYPE,
                                                                 training=self.training,
                                                                 idx=i)

            pooled_feature_maps = utils.GlobalAveragePooling2D(input=pooled_feature_maps,
                                                               n_class=channel_n,
                                                               name='GAP',
                                                               keep_dims=False)

            detector_bbox_refinements = utils.fully_connected('detector_bbox_refinements',
                                                              pooled_feature_maps,
                                                              config.N_CLASS * 4)
            detector_bbox_refinements = tf.reshape(detector_bbox_refinements, (-1, config.TRAIN_ROIS_PER_IMAGE, config.N_CLASS, 4))

            detector_class_logits = utils.fully_connected('detector_class_logits',
                                                          pooled_feature_maps,
                                                          config.N_CLASS)
            detector_class_logits = tf.reshape(detector_class_logits, (-1, config.TRAIN_ROIS_PER_IMAGE, config.N_CLASS))

        return detector_class_logits, detector_bbox_refinements

    def refine_detections_graph(self, rois, probs, deltas, config):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in image coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        # Class IDs per ROI
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = utils.apply_box_deltas_graph2(rois, deltas_specific)

        # Clip boxes to image window
        window = np.array([0, 0, 1, 1], dtype=np.float32) #######################################################
        refined_rois = utils.clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if config.DETECTION_MIN_CONFIDENCE:
            conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
            keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse_tensor_to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)
            # Map indicies
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                             dtype=tf.int64)
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]
        # Keep top detections
        roi_count = config.DETECTION_MAX_INSTANCES
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
        return detections



    def detection_targets_graph(self, proposals, gt_class_ids, gt_boxes, config):
        """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.

        Inputs:
        proposals: [Batch_size, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                Class-specific bbox refinements.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """
        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.squeeze(proposals, axis=0)
            proposals = tf.identity(proposals)

        # Remove zero padding
        proposals, _ = self.trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = self.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")

        # Compute overlaps matrix [proposals, gt_boxes]
        #############################################################
        # overlaps = overlaps_graph(proposals, gt_boxes)
        #
        overlaps = self.overlaps_graph(tf.round(proposals * config.IMG_SIZE[0]),
                                  tf.round(gt_boxes * config.IMG_SIZE[0]))

        # Compute overlaps with crowd boxes [anchors, crowds]
        # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        # crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        # no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine postive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max > 0.)  ################################### >= 0.5
        positive_indices = tf.where(positive_roi_bool)[:, 0]  #####################################################
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(roi_iou_max <= 0.)[:, 0]  ################################## 0.5

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = self.box_refinement_graph2(tf.round(positive_rois * config.IMG_SIZE[0]),
                                       tf.round(roi_gt_boxes * config.IMG_SIZE[0]),
                                       config)
        # deltas /= config.BBOX_STD_DEV ######################################################

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

        return rois, roi_gt_class_ids, deltas

    def overlaps_graph(self, boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeate boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeate() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps

    def trim_zeros_graph(self, boxes, name=None):
        """Often boxes are represented with matricies of shape [N, 4] and
        are padded with zeros. This removes zero boxes.

        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
        """
        non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
        boxes = tf.boolean_mask(boxes, non_zeros, name=name)
        return boxes, non_zeros

    def box_refinement_graph2(self, box, gt_box, config):
        """Compute refinement needed to transform box to gt_box.
        box and gt_box are [N, (y1, x1, y2, x2)]
        """
        box = tf.cast(box, tf.float32)
        gt_box = tf.cast(gt_box, tf.float32)

        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width

        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width

        dy = (gt_center_y - center_y) / config.IMG_SIZE[0] * 100
        dx = (gt_center_x - center_x) / config.IMG_SIZE[0] * 100
        dh = (gt_height - height) / config.IMG_SIZE[0] * 100
        dw = (gt_width - width) / config.IMG_SIZE[0] * 100

        result = tf.stack([dy, dx, dh, dw], axis=1)
        return result

