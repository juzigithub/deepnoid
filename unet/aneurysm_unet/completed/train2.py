'''
Data Load
Model Load
Train
Result Report

train << model, loader
by 신형은 주임
'''
'''
import config as cfg

class Train:
    # class initialize
    def __init__(self):
        self.A = cfg.A
        self.B = cfg.B
        self.batch_size = cfg.BATCH_SIZE

    def train(self):
        pass


if __name__ == '__main__':
    trainer = Train()
    trainer.train()
'''

import tensorlayer as tl
import tensorflow as tf
import time
import os
import numpy as np
import cv2
import performance_eval as pe
# import config as cfg
import model
import loader

os.environ["CUDA_VISIBLE_DEVICES"] = "6"  ####################################### 1 -> 6


class Train:
    # class initialize
    def __init__(self, batch_size, img_size, n_class, depth, n_filter, training_path, init_learning,
                 decay_rate, validation_ratio, decay_step, n_epoch):

        ###########여기를 없애시면 됩니다################
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_class = n_class
        self.depth = depth
        self.n_filter = n_filter
        self.training_path = training_path
        self.init_learning = init_learning
        self.decay_rate = decay_rate
        self.validation_ratio = validation_ratio
        self.decay_step = decay_step
        self.n_epoch = n_epoch
        ############################################

        self.p_eval = pe.performance()
        self.data_loader = loader.DataLoader(img_size=self.img_size)
        self.model = model.Model(img_size=img_size, n_channel=1, n_class=2,
                                 batch_size=self.batch_size, n_filter=self.n_filter, depth=self.depth)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs')

        print('')
        print('>>> Data Loading Started')
        print('')

        dstime = time.time()

        # data_list_loader + img_grey_size 까지 진행 -> pkl 데이터로 저장 -> pkl 데이터 로드
        # 다른 변수와 달리 valX = [abnorm(0) or norm(1) or else(2), img_da1ta] 로 구성
        self.trainX, self.trainY, self.valX, self.valY = self.data_loader.load_data(mode='train')

        detime = time.time()

        print('')
        print('>>> Data Loading Complete. Consumption Time :', detime - dstime)
        print('')

    def optimizer(self, global_step):
        ########################## global_step 빼고 전부 cfg로 바꾸시면 됩니다. #################################
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning,
                                                                     global_step=global_step,
                                                                     decay_steps=self.decay_step,
                                                                     decay_rate=self.decay_rate,
                                                                     staircase=True,
                                                                     name='learning_rate')
        #####################################################################################################

        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.model.loss,
                                                                                                        global_step=global_step)

    def save_yn(self, epoch):
        ######### self.n_epoch 과 epoch % 3 의 3을 cfg로 바꾸시면 됩니다.############
        if epoch == 0 or epoch + 1 == self.n_epoch or epoch % 3 == 0:
            return True
        else:
            return False
        #########################################################################

    def train(self):

        global_step = tf.Variable(0, trainable=False, name='global_step')

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        ########### batch_size 를 cfg로 바꾸시면 됩니다. ##############
        train_step = int(len(self.trainX[1]) / self.batch_size)
        val_step = int(len(self.valX[1]) / self.batch_size)
        ############################################################

        print('>>> Train step:', train_step, 'Validation step:', val_step)
        print('')

        with tf.Session() as sess:

            saver = tf.train.Saver()
            self.writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

            print("BEGIN TRAINING")

            total_training_time = 0

            for epoch in range(self.n_epoch):

                mean_iou_list = []
                unfiltered_iou_list = []
                loss_list = []

                start = time.time()

                total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0

                ######## dropout_rate를 cfg로 바꾸시면 됩니다.#############
                # Iterator 위한 feed_dict
                tr_feed_dict = {self.model.X: self.trainX[1], self.model.Y: self.trainY,
                                self.model.X_ADD: self.trainX[0],
                                self.model.training: True, self.model.drop_rate: 0.2}
                sess.run(self.model.iter.initializer, feed_dict=tr_feed_dict)
                ####################################################

                for _ in range(train_step):
                    cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                    total_cost += cost
                    step += 1

                    print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.n_epoch, 'Step:', step, '/', train_step,
                          '  Batch loss:', cost)

                for _ in range(val_step):

                    # 모델에 validation 데이터를 넣어 줄 Feed Dict입니다.
                    val_feed_dict = {self.model.X: self.valX[1], self.model.Y: self.valY,
                                     self.model.X_ADD: self.valX[0],
                                     self.model.training: False,
                                     self.model.drop_rate: 0}  #################################################
                    # train 데이터와 마찬가지로 iterator를 초기화해줍니다.
                    sess.run(self.model.iter.initializer, feed_dict=val_feed_dict)

                    val_results, predicted_result, x_list, y_list, address = sess.run([self.model.results,
                                                                                       self.model.foreground_predicted,
                                                                                       self.model.features,
                                                                                       self.model.labels,
                                                                                       self.model.address],
                                                                                      feed_dict=val_feed_dict)

                    ious = list(val_results[0])
                    unfiltered_iou = np.mean(ious)

                    iou_list = []

                    for iou in ious:
                        if iou > 0.01:
                            iou_list.append(iou)

                    after_filtered_length = len(iou_list)
                    before_filtered_length = len(ious)

                    if after_filtered_length == 0:
                        mean_iou = 0

                    else:
                        mean_iou = np.mean(iou_list)

                    total_val_iou += mean_iou
                    total_val_unfiltered_iou += unfiltered_iou

                    ############################# 보기 편하도록 이미지 저장 기능을 함수로 바꿔봅시다! #########################################
                    ###### 필요한 함수 : 1) 이미지 경로 설정하는 함수
                    # train.py의 _make_img_full_path 참고
                    # 2) 세 종류의 이미지를 불러와서 expand_dim, squeeze, concat 등 중복된 기능을 한번에 처리하는 함수
                    # train.py의 _masking_rgb 참고
                    # 3) 함수1과 함수2를 내부에 넣고, imwrite까지 수행하는 총괄함수
                    # train.py의 _make_img 참고

                    if self.save_yn(epoch):

                        val_img_save_path = './imgs/' + str(epoch + 1) + '/merged'
                        raw_val_img_save_path = './imgs/' + str(epoch + 1) + '/pred'
                        label_val_img_save_path = './imgs/' + str(epoch + 1) + '/label'

                        tl.files.exists_or_mkdir(val_img_save_path)
                        tl.files.exists_or_mkdir(raw_val_img_save_path)
                        tl.files.exists_or_mkdir(label_val_img_save_path)

                        for img_idx, label in enumerate(predicted_result):
                            # path/abnorm_100_FILE00086.png
                            # ex) address_list[img_idx] = [0, 100, 96]

                            add1, add2, add3 = address[img_idx]
                            full_add = '/{0}_{1}_FILE{2}.png'.format('abnorm' if add1 == 0 else 'norm', add2, add3)

                            val_img_fullpath = val_img_save_path + full_add
                            raw_val_img_fullpath = raw_val_img_save_path + full_add
                            label_val_img_fullpath = label_val_img_save_path + full_add

                            test_image = x_list[img_idx]

                            test_image = np.expand_dims(test_image, axis=0)

                            _, pred_image = cv2.threshold(label, 0.5, 1.0, cv2.THRESH_BINARY)

                            pred_image = np.expand_dims(pred_image, axis=3)
                            pred_image = np.expand_dims(pred_image, axis=0)

                            G = np.zeros([1, 256, 256, 1])
                            B = np.zeros([1, 256, 256, 1])
                            R = pred_image

                            pred_image = np.concatenate((B, G, R), axis=3)

                            pred_image = np.squeeze(pred_image)

                            tR = test_image
                            tG = test_image
                            tB = test_image

                            test_image = np.concatenate((tB, tG, tR), axis=3)
                            test_image = np.squeeze(test_image)

                            label_image = y_list[img_idx][:, :, 0]
                            label_image = np.expand_dims(label_image, axis=0)
                            label_image = np.expand_dims(label_image, axis=3)

                            lR = label_image
                            lG = label_image
                            lB = label_image

                            label_image = np.concatenate((lB, lG, lR), axis=3)
                            label_image = np.squeeze(label_image)

                            label_image = label_image * 255
                            cv2.imwrite(label_val_img_fullpath, label_image)

                            test_image = test_image.astype(float)
                            pred_image = pred_image * 255
                            cv2.imwrite(raw_val_img_fullpath, pred_image)

                            ###################### w, p 도 cfg로 바꾸시면 됩니다. ##########################
                            w = 40
                            p = 0.0001
                            result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0)
                            cv2.imwrite(val_img_fullpath, result * 255)
                #######################################################################################

                if self.save_yn(epoch):  # (epoch+1) % 2 -> epoch % cfg.SAVING_EPOCH
                    tl.files.exists_or_mkdir('./model' + '/' + str(epoch + 1))
                    save_path = './model' + '/' + str(epoch + 1) + '/Unet.ckpt'

                    saver.save(sess, save_path)
                    print(">>> Model SAVED")
                    print('')

                end = time.time()
                training_time = end - start
                total_training_time += training_time

                Loss = total_cost / train_step
                Valdation_IoU = total_val_iou / val_step
                Valdation_Unfiltered_IoU = total_val_unfiltered_iou / val_step

                print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.n_epoch,
                      'Loss =', '{:.4f}  '.format(Loss),
                      'Valdation IoU:{:.4f}   '.format(Valdation_IoU),
                      'Valdation Unfiltered IoU:{:.4f}   '.format(Valdation_Unfiltered_IoU),
                      'Training time: {:.2f}  '.format(training_time))

                result_dict = {self.p_eval.mean_iou: Valdation_IoU,
                               self.p_eval.tot_iou: Valdation_Unfiltered_IoU,
                               self.p_eval.loss: Loss}

                # TB
                summary = sess.run(self.merged_summary, feed_dict=result_dict)
                self.writer.add_summary(summary, global_step=epoch)

                mean_iou_list.append(Valdation_IoU)
                unfiltered_iou_list.append(Valdation_Unfiltered_IoU)
                loss_list.append(Loss)

            print("")
            print("TRAINING COMPLETE")


if __name__ == "__main__":
    data_path = '/home/hshin255/data/BRATS_ORI/training/HGG'
    # data_path = "C:\\Users\\hshin\\Desktop\\sample_files\\BRATS_ORI\\training\\HGG"

    ######### 최종적으로 trainer = Train(안에 아무 것도 들어가지 않게) 만 입력해도 실행되도록 코드 짜시면 됩니다! ###########
    # trainer = Train()
    trainer = Train(batch_size=28,
                    n_epoch=200,
                    img_size=256,
                    n_class=1,
                    depth=4,
                    n_filter=32,
                    training_path=data_path,
                    init_learning=0.005,
                    decay_rate=0.9,
                    validation_ratio=15,
                    decay_step=2500)

    trainer.train()