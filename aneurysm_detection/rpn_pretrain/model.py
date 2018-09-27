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

        ##############################################

        self.rpn_class_logitss, self.rpn_bbox_refinements = self.model()
        self.rpn_class_loss = utils.rpn_class_loss_graph(self.rpn_class_label, self.rpn_class_logitss)
        self.rpn_bbox_loss = utils.rpn_bbox_loss_graph(cfg, self.rpn_bbox_label, self.rpn_class_label, self.rpn_bbox_refinements)

        ############## lambda 값은 변경해야. 일단 0.5로 ##############
        self.loss = 0.5 * self.rpn_class_loss + 0.5 * self.rpn_bbox_loss
        # self.reconstruction_loss = tf.reduce_sum(tf.squared_difference(utils.flatten('logit_flatten', tf.sigmoid(self.logit)),
        #                                                                utils.flatten('X_flatten', tf.sigmoid(self.X))),
        #                                          1)
        # self.latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.gamma) + tf.square(self.mean) - 1 - self.gamma, 1)
        # self.loss = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
        #############################################

    def model(self):

        channel_n = cfg.INIT_N_FILTER

        feature_maps = self.feature_extractor(self.X, channel_n, cfg.PRETRAIN_N_LAYERS, cfg.N_DOWNSAMPLING)
        rpn_feature_maps = tf.expand_dims(feature_maps[tf.shape(feature_maps)[0]//2], axis=0)
        rpn_feature_maps = [rpn_feature_maps]
        rpn_class_logitss, rpn_class_probs, rpn_bbox_refinements = self.rpn_bbox_generator(rpn_feature_maps,
                                                                                cfg.RPN_N_FILTER,
                                                                                len(cfg.ANCHOR_RATIOS))

        self.proposals = self.region_proposal_network(self.anchors, rpn_bbox_refinements, rpn_class_probs, self.training)
        print(self.proposals)

        return rpn_class_logitss, rpn_bbox_refinements


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
                if idx + 1 < n_layer:
                    l = utils.maxpool(name='maxpool_{}'.format(idx),
                                      inputs=l,
                                      pool_size=[2, 2],
                                      strides=[2, 2],
                                      padding='same')
                    channel_n *= 2
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

                shared = utils.residual_block_dw_dr(name='rpn_shared_{}'.format(idx),
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
            boxes = utils.apply_box_deltas_graph2(pre_nms_anchors, deltas_slice, cfg)

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



