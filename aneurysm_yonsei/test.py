import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time
import cv2
import loader
import config as cfg
from model import Model        # choose model
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

# directly making nifti files instead of making 3 npy files

class Test:
    def __init__(self, restore=True):

        self.model = Model()
        self.restore = restore

        if cfg.REBUILD_VAL_DATA:
            print('')
            print('>>> Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_VALID_DATA_PATH)

            loader.save_val_resized_dcm_as_npy(cfg.VAL_DATA_PATH, cfg.SAVE_VALID_DATA_PATH, cfg.VAL_FILE_NAME)

            detime = time.time()

            print('')
            print('>>> Data Saving Complete. Consumption Time :', detime - dstime)
            print('')

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        # self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        # self.model_path = '.{0}model{0}test{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        # self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        # patient_path_list = tl.files.load_folder_list(cfg.VAL_DATA_PATH)
        # self.patient_id_list = [os.path.basename(p) for p in patient_path_list]
        self._make_path()

        self.merged_summary = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter(self.log_path)


    def test(self):
        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            # It's a tensorflow class saving ckpt file
            saver = tf.train.Saver(var_list = tf.global_variables())

            # save graphs from tensorboard
            # self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'weights.ckpt')
                # saver.restore(sess, tf.train.get_checkpoint_state(self.ckpt_path))


            print("BEGIN TESTING")

            test_X = np.load(cfg.SAVE_VALID_DATA_PATH + cfg.VAL_FILE_NAME)
            print('text_X', test_X.shape)
            test_X = test_X.reshape((-1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, 1))
            test_Y = np.concatenate((test_X, test_X), axis=-1)
            img_idx = 0
            # patch_list = []
            for batch in tl.iterate.minibatches(inputs=test_X, targets=test_Y,
                                                batch_size=cfg.N_PATCH_TO_IMG, shuffle=False):
                img_idx += 1
                batch_x, batch_y = batch

                test_feed_dict = {self.model.X: batch_x,
                                 self.model.Y: batch_y,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                pred = sess.run(self.model.pred, feed_dict=test_feed_dict)

                pred = np.argmax(pred, axis=-1)
                pred = np.expand_dims(pred, axis=-1)

                pred_patch_list = utils.reconstruct_from_patches_nd(pred, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1), cfg.PATCH_STRIDE)
                ori_patch_list = utils.reconstruct_from_patches_nd(batch_x, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1), cfg.PATCH_STRIDE)

                # pred_patch_list = np.argmax(pred_patch_list, axis=-1)
                print('pred_patch_list.shape',pred_patch_list.shape)

                num_labels, markers, states, cent = cv2.connectedComponentsWithStats(np.uint8(pred_patch_list))
                print('states', states)
                for state in states:
                    pred_patch_list = cv2.rectangle(np.uint8(pred_patch_list), tuple(state[0:2] - 10), tuple(state[0:2] + state[2:4] + 10), (0, 255, 0), 2)


                pred_patch_list = utils.masking_rgb(pred_patch_list, color='red')
                ori = utils.masking_rgb(ori_patch_list, color=None)

                cv2.imwrite('./img/test/for_nifti/ori_{}.jpg'.format(img_idx), ori)
                cv2.imwrite('./img/test/for_nifti/pred_{}.jpg'.format(img_idx), pred_patch_list + ori)

                # utils.save_array_as_nifty_volume(patch_list, './img/test/for_nifti/{}.nii.gz'.format(img_idx))


            print("TESTING COMPLETED")

    def _make_path(self):
        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir('./img/test/for_nifti/')

if __name__ == "__main__":
    tester = Test(restore=True)
    tester.test()
