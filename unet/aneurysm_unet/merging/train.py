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

import Unet.aneurysm_unet.merging.performance_eval as pe
import Unet.aneurysm_unet.merging.config as cfg
import Unet.aneurysm_unet.merging.model as model
# import Unet.aneurysm_unet.merging.utils as utils
import Unet.aneurysm_unet.merging.loader as loader  ####################


os.environ["CUDA_VISIBLE_DEVICES"] = "6" ####################################### 1 -> 6


class Train:
    # class initialize
    def __init__(self):
        self.p_eval = pe.performance()
        self.data_loader = loader.DataLoader(img_size=cfg.IMG_SIZE)


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


        self.model = model.Model(img_size=cfg.IMG_SIZE,
                                 n_channel=1,
                                 n_class=2,
                                 batch_size=cfg.BATCH_SIZE,
                                 n_filter=cfg.INIT_N_FILTER,
                                 depth=cfg.DEPTH)
        self.p_eval = pe.performance()


        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs')

    def optimizer(self, global_step):
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=cfg.INIT_LEARNING_RATE,
                                                                     global_step=global_step,
                                                                     decay_steps=cfg.DECAY_STEP,
                                                                     decay_rate=cfg.DECAY_RATE,
                                                                     staircase=cfg.DECAY_STAIRCASE,
                                                                     name='learning_rate')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.model.loss,
                                                                                                        global_step=global_step)
    def save_yn(self, epoch):
        if epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0:
            return True
        else :
            return False

    def train(self):

        # 배치정규화를 진행하는 경우 배치정규화의 스탭을 결정하는 변수로 0입니다.
        global_step = tf.Variable(0, trainable=False)

        # 배치정규화를 진행하는 경우 배치별 이동평균과 표준편차를 갱신해주는 update operation을 실행하고 지정해줍니다.
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        # 각각의 전체 데이터셋을 배치사이즈로 나누어 한 에폭당 몇 스텝이 진행되는가 계산합니다.
        train_step = int(len(self.trainX) / cfg.BATCH_SIZE)
        val_step = int(len(self.valX) / cfg.BATCH_SIZE)

        print('>>> Train step:', train_step, 'Validation step:', val_step)
        print('')

        # 텐서플로 그래프가 생성되고 작업이 수행되는 세션을 선언해줍니다.
        with tf.Session() as sess:

            # 모델을 저장하는 것은 모델의 가중치 같은 변수들을 저장하는 것인데 이를 텐서플로에선 ckpt, 체크포인트 파일이라고 합니다. ckpt 파일을 저장하는 텐서플로 클래스입니다.
            saver = tf.train.Saver()

            # 텐서보드에 생성한 모델의 그래프를 저장해줍니다.
            self.writer.add_graph(sess.graph)

            # 세션상의 글로벌 변수 초기화를 진행합니다. 변수 초기값은 개별 변수별로 지정해줘야합니다.
            sess.run(tf.global_variables_initializer())

            ####################################################
            # Iterator 위한 feed_dict
            tr_feed_dict = {self.model.X: self.trainX, self.model.Y: self.trainY,
                            self.model.training: True, self.model.drop_rate: cfg.DROPOUT_RATE}
            sess.run(self.model.iter.initializer, feed_dict=tr_feed_dict)
            ####################################################



            print("BEGIN TRAINING")

            total_training_time = 0

            # ==========================  Traing start  =========================================#
            # 전달 받은 epoch 수 만큼 학습을 진행하는 loop 문 입니다.
            for epoch in range(cfg.EPOCHS):

                mean_iou_list = []
                unfiltered_iou_list = []
                loss_list = []

                start = time.time()

                total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0


                # 한 에폭마다 학습하는 각 개별 스텝을 진행하는 loop 문 입니다.
                # shuffles every batch and iterates
                # for batch in tl.iterate.minibatches(inputs=self.trainX, targets=self.trainY,
                #                                     batch_size=self.batch_size, shuffle=True):
                # 경로 내의 학습데이터를 마찬가지로 랜덤 셔플해줍니다. 경로 셔플과 마찬가지의 효과를 줍니다.
                # trainX, trainY = self.data_loader.data_shuffle(self.trainX, self.trainY)

                # 한 에폭마다 학습하는 각 개별 스텝을 진행하는 loop 문 입니다.
                # for batch in range(train_step):
                    # 데이터모듈을 이용하여 매 스탭마다 학습에 사용할 데이터 경로를 불러오며 스텝이 진행되면 다음 배치데이터를 불러옵니다.
                    # batch_xs_list, batch_ys_list = self.data_loader.next_batch(data_list=trainX, label=trainY,
                    #                                                            idx=batch,
                    #                                                            batch_size=cfg.BATCH_SIZE)

                    # 데이터모듈을 이용하여 위에서 불러온 데이터 경로에서 이미지 데이터를 읽어서 배치데이터를 만듭니다.
                    # batch_xs = self.data_loader.read_image_grey_resized(batch_xs_list)
                    # batch_ys = self.data_loader.read_label_grey_resized(batch_ys_list)

                    # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                    # tr_feed_dict = {self.model.X: batch_xs, self.model.Y: batch_ys, self.model.training: True,
                    #                 self.model.drop_rate: 0.2}

                    # 모델을 위에서 선언한 session을 run 시켜서 학습시키고 결과물로 cost값을 받습니다.
                    # cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)
                    #
                    # total_cost += cost
                    # step += 1
                    #
                    # 학습 과정에서의 현재 에폭과 스텝 그리고 배치 Loss 값을 출력합니다.
                    # print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % cfg.EPOCHS, 'Step:', step, '/', train_step,
                    #       '  Batch loss:', cost)

                for _ in range(train_step):
                    cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                    total_cost += cost
                    step += 1

                    # 학습 과정에서의 현재 에폭과 스텝 그리고 배치 Loss 값을 출력합니다.
                    print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % cfg.EPOCHS, 'Step:', step, '/', train_step,
                          '  Batch loss:', cost)



                for _ in range(val_step):


                    #
                    # # 데이터모듈을 이용하여 매 스탭마다 밸리데이션에 사용할 데이터 경로를 불러오며 스텝이 진행되면 다음 배치데이터를 불러옵니다.
                    # val_batch_xs_list, val_batch_ys_list = self.data_loader.next_batch(data_list=self.valX,
                    #                                                                    label=self.valY, idx=batch,
                    #                                                                    batch_size=cfg.BATCH_SIZE)
                    #
                    # # 데이터모듈을 이용하여 위에서 불러온 데이터 경로에서 이미지 데이터를 읽어서 배치데이터를 만듭니다.
                    # val_batch_xs = self.data_loader.read_image_grey_resized(val_batch_xs_list)
                    # val_batch_ys = self.data_loader.read_label_grey_resized(val_batch_ys_list)
                    #
                    # # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                    # val_feed_dict = {self.model.X: val_batch_xs, self.model.Y: val_batch_ys,
                    #                  self.model.training: False, self.model.drop_rate: 0}
                    #
                    # # 밸리데이션 결과 IoU(Intersection of Union)을 계산합니다. Image Segmentation에선 IoU를 보통 Accuracy로 사용합니다.
                    # # model.iou에선 [acc, mean_iou, unfiltered_iou]를 리턴합니다.
                    # val_results, predicted_result = sess.run([self.model.results, self.model.foreground_predicted],
                    #                                          feed_dict=val_feed_dict)
                    # # acc, val_mean_iou, val_unfiltered_iou = val_results




                    # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                    val_feed_dict = {self.model.X: self.valX[1], self.model.Y: self.valY, self.model.X_ADD: self.valX[0],
                                     self.model.training: False, self.model.drop_rate: 0}  #################################################

                    # 밸리데이션 결과 IoU(Intersection of Union)을 계산합니다. Image Segmentation에선 IoU를 보통 Accuracy로 사용합니다.
                    # model.iou에선 [acc, mean_iou, unfiltered_iou]를 리턴합니다.
                    val_results, predicted_result, x_list, y_list, address = sess.run([self.model.results,
                                                                                       self.model.foreground_predicted,
                                                                                       self.model.X,
                                                                                       self.model.Y,
                                                                                       self.model.X_ADD],
                                                                                      feed_dict=val_feed_dict)
                    # acc, val_mean_iou, val_unfiltered_iou = val_results


                    # 받은 배치 IoU값을 리스트로 변환합니다.
                    ious = list(val_results[0])
                    unfiltered_iou = np.mean(ious)


                    # IoU가 0.01 이상, 즉 일정이상 예측해낸 IoU 값들만 모아서 진단 정확도와 질병으로 판단 했을 때의 IoU 값을 따로 계산합니다.
                    iou_list = []

                    for iou in ious:
                        if iou > 0.01:
                            iou_list.append(iou)

                    after_filtered_length = len(iou_list)
                    before_filtered_length = len(ious)

                    # val_batch_acc = after_filtered_length / before_filtered_length

                    if after_filtered_length == 0: ####### len(iou_list) -> after_filtered_length
                        mean_iou = 0

                    else:
                        mean_iou = np.mean(iou_list)

                    # 배치별 IoU값과 정확도를 전체 IoU값과 정확도에 더합니다. 에폭이 종료되면 평균 IoU와 평균 정확도로 환산합니다.
                    total_val_iou += mean_iou
                    total_val_unfiltered_iou += unfiltered_iou

#################################### 함수로 #########################################
                    # 학습 시작 에폭과 끝에폭 그리고 saving epoch의 배수마다 이미지와 모델을 저장하게 합니다.
                    # if epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0 : # (epoch+1) % 2 -> epoch % cfg.SAVING_EPOCH


                    if self.save_yn(epoch) :

                        # 밸리데이션 결과 이미지를 저장하는 경로입니다.
                        # val_img_save_path 는 학습이미지(원본이미지)와 예측이미지를 Overlap 시켜 환부에 마스크 이미지를 씌워주며
                        # raw_val_img_save_path는 예측이미지를, label_val_img_save_path는 라벨이미지를 저장하는 경로입니다.
                        # 윈도우에서 실행시키려면 /를 \\로 교체해야합니다.
                        val_img_save_path = './imgs/' + str(epoch + 1) + '/merged'
                        raw_val_img_save_path = './imgs/' + str(epoch + 1) + '/pred'
                        label_val_img_save_path = './imgs/' + str(epoch + 1) + '/label'

                        # 각 개별 경로가 존재하는지 확인하고 없는 경우 경로를 생성합니다.
                        tl.files.exists_or_mkdir(val_img_save_path)
                        tl.files.exists_or_mkdir(raw_val_img_save_path)
                        tl.files.exists_or_mkdir(label_val_img_save_path)

                        # 예측된 배치 결과를 loop하면서 개별 이미지로 저장하는 loop문입니다.

                        for img_idx, label in enumerate(predicted_result):
                            # 각 이미지 종류별 이미지를 저장하는 절대경로입니다. 이미지 파일명은 '밸리데이션 index'_'이미지 번호'.png 로 저장됩니다.
                            # path/abnorm_100_FILE00086.png
                            # ex) address_list[img_idx] = [0, 100, 96]

                            add1, add2, add3 = address[img_idx]
                            full_add = '/{0}_{1}_FILE{2}.png'.format('abnorm' if add1 == 0 else 'norm', add2, add3)

                            val_img_fullpath = val_img_save_path + full_add
                            raw_val_img_fullpath = raw_val_img_save_path + full_add
                            label_val_img_fullpath = label_val_img_save_path + full_add

                            # 라벨이미지를 가져옵니다.
                            test_image = x_list[img_idx]

                            # 라벨이미지 저장을 위해 3채널 RGB 데이터가 필요하고 배치 차원을 맞춰주기 위해 차원확장을 진행합니다.
                            # 이미지의 차원은 현재 [B, H, W, C] 로 배치, 세로, 가로, 채널로 되어있습니다.
                            test_image = np.expand_dims(test_image, axis=0)

                            # 예측 결과를 threshold(기준 값을 경계로 0과 1 바이너리화를 진행합니다.)
                            # 사용법 : _, img = cv2.threshold(이미지, 경계값, 바이너리최대값, 바이너리옵션)으로
                            # 옵션을 cv2.THRESH_BINARY로 진행하면 검은색 흰색 이미지가, cv2.THRESH_BINARY_INV로 진행하면 흰색 검은색 이미지가 저장됩니다.
                            # 자세한 사항은 cv2 홈페이지를 참조하세요.
                            _, pred_image = cv2.threshold(label, 0.5, 1.0, cv2.THRESH_BINARY)

                            # cv2의 결과는 2차원(H, W) 입니다. 따라서 마찬가지로 0차원과 4차원에 차원을 덧대주어서 차원을 맞춰줍니다.
                            pred_image = np.expand_dims(pred_image, axis=3)
                            pred_image = np.expand_dims(pred_image, axis=0)

                            # 예측이미지의 마스크 색을 결정합니다.
                            # 예측이미지값을 R에 넣으면 빨간 마스킹 이미지가, B에 넣으면 파란 마스킹 이미지가, G에 넣으면 녹색 마스킹 이미지가 생성됩니다.
                            G = np.zeros([1, 256, 256, 1])
                            B = np.zeros([1, 256, 256, 1])
                            R = pred_image

                            # R, G, B 채널을 concat 해서 하나의 차원에 정렬해줍니다.
                            pred_image = np.concatenate((B, G, R), axis=3)

                            # 필요없는 차원을 squeeze 해줍니다.
                            pred_image = np.squeeze(pred_image)

                            # test_image는 원본이 그대로 필요하므로 R, G, B 모두에 데이터를 넣어줍니다.
                            tR = test_image
                            tG = test_image
                            tB = test_image

                            # 위 concat, squeeze와 동일합니다
                            test_image = np.concatenate((tB, tG, tR), axis=3)
                            test_image = np.squeeze(test_image)

                            # 위 과정을 label_img도 동일하게 진행해줍니다.
                            label_image = y_list[img_idx][:, :, 0]
                            label_image = np.expand_dims(label_image, axis=0)
                            label_image = np.expand_dims(label_image, axis=3)

                            lR = label_image
                            lG = label_image
                            lB = label_image

                            label_image = np.concatenate((lB, lG, lR), axis=3)
                            label_image = np.squeeze(label_image)

                            # 바이너리화된 이미지는 (0, 1)의 데이터 이므로 RGB로 변경하려면 255를 곱해주어야 합니다.
                            label_image = label_image * 255
                            cv2.imwrite(label_val_img_fullpath, label_image)

                            # 위와 동일합니다.
                            test_image = test_image.astype(float)
                            pred_image = pred_image * 255
                            cv2.imwrite(raw_val_img_fullpath, pred_image)

                            # 원본이미지에 예측결과를 마스킹해줍니다.
                            # 마스킹 비율을 결정하는 파라메터가 w이고 각 이미지의 적용비율은 p로 결정합니다. w와 p를 바꿔가면서 저장하며 가시성 좋은 값을 찾으면 됩니다.
                            w = 40
                            p = 0.0001
                            result = cv2.addWeighted(pred_image, float(100 - w) * p, test_image, float(w) * p, 0)
                            cv2.imwrite(val_img_fullpath, result * 255)


                ################### 저장 ##################
                if self.save_yn(epoch):  # (epoch+1) % 2 -> epoch % cfg.SAVING_EPOCH
                    # 모델을 저장할 경로를 확인하고 없으면 만들어줍니다.
                    tl.files.exists_or_mkdir('./model' + '/' + str(epoch + 1))

                    # 모델 저장을 위한 절대경로입니다. '파일명'.ckpt로 저장합니다.
                    save_path = './model' + '/' + str(epoch + 1) + '/Unet.ckpt'

                    # 모델을 저장합니다.
                    saver.save(sess, save_path)
                    print(">>> Model SAVED")
                    print('')

                end = time.time()
                training_time = end - start
                total_training_time += training_time

                Loss = total_cost / train_step
                Valdation_IoU = total_val_iou / val_step
                Valdation_Unfiltered_IoU = total_val_unfiltered_iou / val_step

                print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % cfg.EPOCHS,
                      'Loss =', '{:.4f}  '.format(Loss),
                      'Valdation IoU:{:.4f}   '.format(Valdation_IoU),
                      'Valdation Unfiltered IoU:{:.4f}   '.format(Valdation_Unfiltered_IoU),
                      'Training time: {:.2f}  '.format(training_time))

                result_dict = {self.p_eval.mean_iou: Valdation_IoU, self.p_eval.tot_iou: Valdation_Unfiltered_IoU,
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
    # data_path = '/home/hshin255/data/BRATS_ORI/training/HGG'
    # data_path = "C:\\Users\\hshin\\Desktop\\sample_files\\BRATS_ORI\\training\\HGG"
    
    

    trainer = Train()

    trainer.train()