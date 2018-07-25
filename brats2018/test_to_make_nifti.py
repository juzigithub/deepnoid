import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time
import loader
import config as cfg
from model import Model
import utils


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU


class Test:
    def __init__(self, restore=True):

        self.model = Model()
        self.restore = restore

        if cfg.REBUILD_TASK1_VAL_DATA:
            print('')
            print('>>> Task1 Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_VALID_DATA_PATH)

            loader.data_saver([cfg.VAL_DATA_PATH], cfg.SAVE_VALID_DATA_PATH, cfg.SPLITS, train=False, shuffle=False)

            detime = time.time()

            print('')
            print('>>> Task1 Data Saving Complete. Consumption Time :', detime - dstime)
            print('')

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        self.model_path = '.{0}model{0}test{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        patient_path_list = tl.files.load_folder_list(cfg.VAL_DATA_PATH)
        self.patient_id_list = [os.path.basename(p) for p in patient_path_list]
        self._make_path()

        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path)


    def test(self):
        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            # It's a tensorflow class saving ckpt file
            saver = tf.train.Saver()

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'brats.ckpt')
                # saver.restore(sess, tf.train.get_checkpoint_state(self.ckpt_path))


            print("BEGIN TESTING")

            test_X = np.load(cfg.SAVE_VALID_DATA_PATH + 'brats_val_image.npy')

            img_idx = 0
            img_list = []
            save_idx = 0
            for batch in tl.iterate.minibatches(inputs=test_X, targets=test_X,
                                                batch_size=cfg.BATCH_SIZE, shuffle=False):
                img_idx += 1
                batch_x, _ = batch

                test_feed_dict = {self.model.X: batch_x,
                                 self.model.Y: batch_x,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                pred = sess.run(self.model.pred, feed_dict=test_feed_dict)
                pred = np.argmax(pred, axis=-1)
                pred[pred == 3] = 4.
                pred.astype(np.float32)

                img_list.append(pred.tolist())

                if img_idx == ((cfg.N_PATCH_TO_IMG * 150) // cfg.BATCH_SIZE) :
                    img_list = np.array(img_list).reshape([-1, 192, 160])
                    img_list = utils.reconstruct_from_patches_nd(img_list, (cfg.IMG_SIZE, cfg.IMG_SIZE), cfg.PATCH_STRIDE)
                    img_list = np.transpose(img_list, [2,1,0])
                    zero_padded = np.pad(img_list, ((41, 39), (30, 18), (3, 2)), 'constant')
                    # zero_padded = np.flip(zero_padded,0)
                    # zero_padded = np.flip(zero_padded,1)

                    utils.save_array_as_nifty_volume(zero_padded, './img/test/for_nifti/{}.nii.gz'.format(self.patient_id_list[save_idx]))
                    # np.save('./img/test/for_nifti/{}.npy'.format(self.patient_id_list[save_idx]), zero_padded)
                    img_list = []
                    img_idx = 0
                    save_idx += 1

            print("TESTING COMPLETED")

    def _make_path(self):
        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir('./img/test/for_nifti/')

if __name__ == "__main__":
    tester = Test(restore=True)
    tester.test()
