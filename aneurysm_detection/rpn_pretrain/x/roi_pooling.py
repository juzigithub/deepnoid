# import sonnet as snt
import tensorflow as tf

# Types of RoI "pooling"
# CROP = 'crop'
# ROI_POOLING = 'roi_pooling'
#
# #
# # class ROIPoolingLayer(snt.AbstractModule):
# #     """ROIPoolingLayer applies ROI Pooling (or tf.crop_and_resize).
# #     RoI pooling or RoI extraction is used to extract fixed size features from a
# #     variable sized feature map using variabled sized bounding boxes. Since we
# #     have proposals of different shapes and sizes, we need a way to transform
# #     them into a fixed size Tensor for using FC layers.
# #     There are two basic ways to do this, the original one in the FasterRCNN's
# #     paper is RoI Pooling, which as the name suggests, it maxpools directly from
# #     the region of interest, or proposal, into a fixed size Tensor.
# #     The alternative way uses TensorFlow's image utility operation called,
# #     `crop_and_resize` which first crops an Tensor using a normalized proposal,
# #     and then applies extrapolation to resize it to the desired size,
# #     generating a fixed size Tensor.
# #     Since there isn't a std support implemenation of RoIPooling, we apply the
# #     easier but still proven alternatve way.
# #     """
# #     def __init__(self, config, debug=False, name='roi_pooling'):
# #         super(ROIPoolingLayer, self).__init__(name=name)
# #         self._pooling_mode = config.pooling_mode.lower()
# #         self._pooled_width = config.pooled_width
# #         self._pooled_height = config.pooled_height
# #         self._pooled_padding = config.padding
# #         self._debug = debug
#
# def _get_bboxes(roi_proposals, im_shape):
#     """
#     Gets normalized coordinates for RoIs (between 0 and 1 for cropping)
#     in TensorFlow's order (y1, x1, y2, x2).
#     Args:
#         roi_proposals: A Tensor with the bounding boxes of shape
#             (total_proposals, 5), where the values for each proposal are
#             (x_min, y_min, x_max, y_max).
#         im_shape: A Tensor with the shape of the image (height, width).
#     Returns:
#         bboxes: A Tensor with normalized bounding boxes in TensorFlow's
#             format order. Its should is (total_proposals, 4).
#     """
#     with tf.name_scope('get_bboxes'):
#         im_shape = tf.cast(im_shape, tf.float32)
#
#         x1, y1, x2, y2 = tf.unstack(
#             roi_proposals, axis=1
#         )
#
#         x1 = x1 / im_shape[1]
#         y1 = y1 / im_shape[0]
#         x2 = x2 / im_shape[1]
#         y2 = y2 / im_shape[0]
#
#         bboxes = tf.stack([y1, x1, y2, x2], axis=1)
#
#         return bboxes
#
# def _roi_crop(roi_proposals, conv_feature_map, im_shape):
#     # Get normalized bounding boxes.
#     bboxes = _get_bboxes(roi_proposals, im_shape)
#     # Generate fake batch ids
#     bboxes_shape = tf.shape(bboxes)
#     batch_ids = tf.zeros((bboxes_shape[0], ), dtype=tf.int32)
#     # Apply crop and resize with extracting a crop double the desired size.
#     crops = tf.image.crop_and_resize(
#         conv_feature_map, bboxes, batch_ids,
#         [pooled_width * 2, pooled_height * 2], name="crops"
#     )
#
#     # Applies max pool with [2,2] kernel to reduce the crops to half the
#     # size, and thus having the desired output.
#     outputs = tf.nn.max_pool(
#             crops, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#             padding='same'
#         ),
#
#     return outputs
# #######################################################################
# # def log2_graph(x):
# #     """Implementatin of Log2. TF doesn't have a native implemenation."""
# #     return tf.log(x) / tf.log(2.0)
#
#
# # KE.Layer -> tf.layers.Layer
# class PyramidROIAlign(tf.layers.Layer):
#     """Implements ROI Pooling on multiple levels of the feature pyramid.
#
#     Params:
#     - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
#
#     Inputs:
#     - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
#              coordinates. Possibly padded with zeros if not enough
#              boxes to fill the array.
#     - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
#     - Feature maps: List of feature maps from different levels of the pyramid.
#                     Each is [batch, height, width, channels]
#
#     Output:
#     Pooled regions in the shape: [batch, num_boxes, height, width, channels].
#     The width and height are those specific in the pool_shape in the layer
#     constructor.
#     """
#
#     def __init__(self, pool_shape, **kwargs):
#         super(PyramidROIAlign, self).__init__(**kwargs)
#         self.pool_shape = tuple(pool_shape)
#
#     def call(self, inputs):
#         # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
#         boxes = inputs[0]
#
#         # Image meta
#         # Holds details about the image. See compose_image_meta()
#         # image_meta = inputs[1]
#
#         # Feature Maps. List of feature maps from different level of the
#         # feature pyramid. Each is [batch, height, width, channels]
#         feature_maps = inputs[2:]
#
#         # Assign each ROI to a level in the pyramid based on the ROI area.
#         y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
#         h = y2 - y1
#         w = x2 - x1
#         # Use shape of first image. Images in a batch must have the same size.
#         # image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
#         # Equation 1 in the Feature Pyramid Networks paper. Account for
#         # the fact that our coordinates are normalized here.
#         # e.g. a 224x224 ROI (in pixels) maps to P4
#         # k = k0 + log2(√wh/224)
#         # image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
#         roi_level = log2_graph(tf.sqrt(h * w))
#         roi_level = tf.minimum(5, tf.maximum(
#             2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
#         roi_level = tf.squeeze(roi_level, 2)
#
#         # Loop through levels and apply ROI pooling to each. P2 to P5.
#         pooled = []
#         box_to_level = []
#         for i, level in enumerate(range(2, 6)):
#             ix = tf.where(tf.equal(roi_level, level))
#             level_boxes = tf.gather_nd(boxes, ix)
#
#             # Box indicies for crop_and_resize.
#             box_indices = tf.cast(ix[:, 0], tf.int32)
#
#             # Keep track of which box is mapped to which level
#             box_to_level.append(ix)
#
#             # Stop gradient propogation to ROI proposals
#             level_boxes = tf.stop_gradient(level_boxes)
#             box_indices = tf.stop_gradient(box_indices)
#
#             # Crop and Resize
#             # From Mask R-CNN paper: "We sample four regular locations, so
#             # that we can evaluate either max or average pooling. In fact,
#             # interpolating only a single value at each bin center (without
#             # pooling) is nearly as effective."
#             #
#             # Here we use the simplified approach of a single value per bin,
#             # which is how it's done in tf.crop_and_resize()
#             # Result: [batch * num_boxes, pool_height, pool_width, channels]
#             pooled.append(tf.image.crop_and_resize(
#                 feature_maps[i], level_boxes, box_indices, self.pool_shape,
#                 method="bilinear"))
#
#         # Pack pooled features into one tensor
#         pooled = tf.concat(pooled, axis=0)
#
#         # Pack box_to_level mapping into one array and add another
#         # column representing the order of pooled boxes
#         box_to_level = tf.concat(box_to_level, axis=0)
#         box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
#         box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
#                                  axis=1)
#
#         # Rearrange pooled features to match the order of the original boxes
#         # Sort box_to_level by batch then box index
#         # TF doesn't have a way to sort by two columns, so merge them and sort.
#         sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
#         ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
#             box_to_level)[0]).indices[::-1]
#         ix = tf.gather(box_to_level[:, 2], ix)
#         pooled = tf.gather(pooled, ix)
#
#         # Re-add the batch dimension
#         pooled = tf.expand_dims(pooled, 0)
#         return pooled
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)

#########################################


def roi_pooling(roi_proposals, conv_feature_map, pooled_shape, feature_pyramid=False):
    '''
    :param roi_proposals: [batch, num_boxes, (y1, x1, y2, x2)]
    :param conv_feature_map: [p2, p3, p4, p5 ..]
    :param pooled_shape: [pooled_width, pooled_height]
    :param feature_pyramid: conv_feature_map is feature_pyramid -> True
    :return: [batch, num_boxes, height, width, channels]
    '''

    def log2_graph(x):
        return tf.log(x) / tf.log(2.0)

    if feature_pyramid:
        n_feature = len(conv_feature_map)

        y1, x1, y2, x2 = tf.split(roi_proposals, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # roi_level = 4 + tf.cast(tf.round(log2_graph(tf.sqrt(h * w))), tf.int32)
        # # 2 <= roi_level <= 5
        # roi_level = tf.minimum(5, tf.maximum(2, roi_level))
        # roi_level = tf.squeeze(roi_level, 2)

        roi_level = n_feature - 2 + tf.cast(tf.round(log2_graph(tf.sqrt(h * w))), tf.int32)
        # 0 <= roi_level <= n_feature-1
        roi_level = tf.minimum(n_feature - 1, tf.maximum(0, roi_level))
        roi_level = tf.squeeze(roi_level, 2)


        pooled = []
        box_to_level = []

        # for i, level in enumerate(range(2, 6)):
        #     ix = tf.where(tf.equal(roi_level, level))
        #     level_boxes = tf.gather_nd(roi_proposals, ix)
        #
        #     # Box indicies for crop_and_resize.
        #     box_indices = tf.cast(ix[:, 0], tf.int32)
        #
        #     # Keep track of which box is mapped to which level
        #     box_to_level.append(ix)
        #
        #     # Stop gradient propogation to ROI proposals
        #     level_boxes = tf.stop_gradient(level_boxes)
        #     box_indices = tf.stop_gradient(box_indices)
        #
        #     # Crop and Resize
        #     # From Mask R-CNN paper: "We sample four regular locations, so
        #     # that we can evaluate either max or average pooling. In fact,
        #     # interpolating only a single value at each bin center (without
        #     # pooling) is nearly as effective."
        #     # Result: [batch * num_boxes, pool_height, pool_width, channels]
        #     pooled.append(tf.image.crop_and_resize(conv_feature_map[i], level_boxes, box_indices, pooled_shape, method="bilinear"))
        for i in range(n_feature):
            ix = tf.where(tf.equal(roi_level, i))
            level_boxes = tf.gather_nd(roi_proposals, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(conv_feature_map[i], level_boxes, box_indices, pooled_shape, method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)

    else :
        box_indices = tf.zeros((roi_proposals[0] * roi_proposals[1],), dtype=tf.int32)
        roi_proposals = tf.squeeze(roi_proposals, axis=0)
        pooled = tf.image.crop_and_resize(conv_feature_map, roi_proposals, box_indices, pooled_shape, method="bilinear")

    print(pooled)
    return pooled