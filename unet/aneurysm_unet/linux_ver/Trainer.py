"""
학습모듈로 메인모듈에서 옵션값들을 전달받아 데이터로더와 모델을 불러와서 실질적으로 학습을 진행하며 학습 중간에 밸리데이션을 진행하여 얼마나 학습이 진행되었는가 체크할 수 있습니다.

- 텐서보드로 모델과 성능을 확인해 볼 수 있습니다.

- 텐서보드 구동은 윈도우 기준 Anaconda prompt에서 tensorboard --logdir=로그파일경로 로 실행한 뒤 화면에 뜨는 주소창으로 접속하면 확인 할 수 있습니다.
  ex) tensorboard --logdir=C:\imsi
"""


import tensorflow as tf
import loader
import unet

import time
import os
import cv2
import numpy as np
import performance_eval as pe
import config as cfg



# GPU가 여러개인경우 특정 GPU에서만 작업을 진행 할 수 있게 GPU 환경을 고정해줍니다.
# GPU가 n개일 경우 순차적으로 0번부터 n-1번까지 존재합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class Trainer:
    def __init__(self, model_root_channel, upsampling_mode, normalization_mode, training_data_path, validation_data_path, pkl_data_path, model_path, downsampling_option,
                 validation_percentage, initial_learning_rate, decay_step, decay_rate, epoch, img_size, n_class, batch_size, saving_epoch, loss, groupnormalization_n, channel_mode,
                 activation):

        # main.py에서 전달 받은 옵션 값
        self.training_path = training_data_path
        self.validation_path = validation_data_path
        self.pkl_data_path = pkl_data_path
        self.model_path = model_path
        self.val_data_cnt = validation_percentage
        self.init_learning = initial_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.epoch_num = epoch
        self.batch_size = batch_size
        self.upsampling_mode = upsampling_mode
        self.normalization_mode = normalization_mode
        self.activation = activation
        self.saving_epoch = saving_epoch
        self.loss = loss
        self.group_n = groupnormalization_n
        self.channel_mode = channel_mode
        self.downsampling_option = downsampling_option
        self.label_channel = n_class + 1
        self.model_root_channel = model_root_channel


        # 데이터로더 모듈 initialize
        self.data_loader = loader.DataLoader(img_size=img_size)

        print('')
        print('')
        print('')
        print('>>> Data Loading Started')
        print('')
        dstime = time.time()

        # 데이터로더 모듈로 학습데이터와 라벨데이터의 경로리스트와 데이터셋 개수를 가져옵니다
        self.trainX, self.trainY, self.valX, self.valY = self.data_loader.load_data(mode='train')
        # self.trainX, self.trainY, _ = self.data_loader.data_list_load(cfg.TRAIN_DATA_PATH,mode='train')
        # print(np.shape(self.trainX))
        # print(self.trainX)

        detime = time.time()
        print('>>> Data Loading Complete. Consumption Time :', detime - dstime)
        print('')
        print('>>> Dataset Split Started')
        print('')
        dsstime = time.time()
        dsetime = time.time()
        print('>>> Train Dataset Count:', len(self.trainX), 'valdation Dataset Count:', len(self.valX[1])) ######################################
        print('')
        print('>>> Data Split Complete. Consumption Time :', dsetime - dsstime)
        print('')

        # 모델 모듈을 initialize 시키고 필요한 옵션값들을 전달 해 줍니다.
        #  n_channel은 학습데이터(X data)의 이미지 채널수를 입력합니다. 보통 Gray scaling을 하면 1채널, 안하면 RGB로 3채널입니다.
        self.model = unet.Model(upsampling_mode=self.upsampling_mode, normalization_mode=self.normalization_mode,
                                img_size=img_size, n_channel=1, n_class=self.label_channel,
                                batch_size=self.batch_size, model_root_channel=self.model_root_channel, loss=self.loss,
                                groupnormalization_n=self.group_n, channel_mode=self.channel_mode,
                                activation=self.activation,
                                downsampling_option=self.downsampling_option)
        self.p_eval = pe.performance()

        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs')

    # 학습을 진행하는 옵티마이저입니다. Adaptive Momentum(Adam) 방식을 사용합니다.
    def optimizer(self, global_step):
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning,
                                                                     global_step=global_step,
                                                                     decay_steps=self.decay_step,
                                                                     decay_rate=self.decay_rate, staircase=True,
                                                                     name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=exponential_decay_learning_rate).minimize(self.model.loss,
                                                                                                        global_step=global_step)

    # 학습을 진행하는 메소드입니다.
    def train(self):

        # 배치정규화를 진행하는 경우 배치정규화의 스탭을 결정하는 변수로 0입니다.
        global_step = tf.Variable(0, trainable=False)

        # 배치정규화를 진행하는 경우 배치별 이동평균과 표준편차를 갱신해주는 update operation을 실행하고 지정해줍니다.
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        # 각각의 전체 데이터셋을 배치사이즈로 나누어 한 에폭당 몇 스텝이 진행되는가 계산합니다.
        train_step = int(len(self.trainX) / self.batch_size)
        val_step = int(len(self.valX[1]) / self.batch_size) #############################################

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
            # saver.restore(sess, '/home/bjh/new_work/180424/dice,lr0.004,dr0.9,epoch200,batch32,channel32,resize,lrelu,NHWC.resizepool_neighbor-neighbor/init_model/Unet.ckpt')
            # print(">>> Model Loaded")
            tr_feed_dict = {self.model.X: self.trainX, self.model.Y: self.trainY, self.model.training: True,
                            self.model.drop_rate: 0.2, }
            sess.run(self.model.iter.initializer, feed_dict=tr_feed_dict)




            print("BEGIN TRAINING")

            total_training_time = 0

            # 전달 받은 epoch 수 만큼 학습을 진행하는 loop 문 입니다.
            for epoch in range(self.epoch_num):

                # 최대수치 체크
                acc_list = []
                mean_iou_list = []
                unfiltered_iou_list = []
                loss_list = []

                # 컨퓨전 매트릭스
                confusion_list = []

                start = time.time()

                total_cost = 0
                total_val_iou = 0
                total_val_acc = 0
                total_val_unfiltered_iou = 0
                step = 0

                # 모델에 데이터를 넣어 줄 Feed Dict입니다.


                # 모델을 위에서 선언한 session을 run 시켜서 학습시키고 결과물로 cost값을 받습니다.
                for _ in range(train_step):
                    cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                    total_cost += cost
                    step += 1

                    # 학습 과정에서의 현재 에폭과 스텝 그리고 배치 Loss 값을 출력합니다.
                    print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.epoch_num, 'Step:', step, '/', train_step,
                          '  Batch loss:', cost)


                # 한 에폭마다 학습이 완료되고 해당 모델로 밸리데이션을 진행하는 loop 문 입니다.
                for _ in range(val_step):
                    # 모델에 데이터를 넣어 줄 Feed Dict입니다.
                    val_feed_dict = {self.model.X: self.valX[1], self.model.Y: self.valY, self.model.training: False,
                                     self.model.drop_rate: 0} #################################################

                    # 밸리데이션 결과 IoU(Intersection of Union)을 계산합니다. Image Segmentation에선 IoU를 보통 Accuracy로 사용합니다.
                    # model.iou에선 [acc, mean_iou, unfiltered_iou]를 리턴합니다.
                    val_results, predicted_result = sess.run([self.model.results, self.model.foreground_predicted],
                                                             feed_dict=val_feed_dict)
                    # acc, val_mean_iou, val_unfiltered_iou = val_results

                    # 받은 배치 IoU값을 리스트로 변환합니다.
                    ious = list(val_results[0])
                    accs = list(val_results[1])
                    # print(accs)  # [ 0.  5.  2. 15.  0.  7.  0. 12.  0.  5.  8.  0. 17.  0.  0. 10.  0. 29. 0.  9. 11.  0.  0. 13.  0. 15. 22.  3. 14.  0.]

                    # 비정상-오브젝트있음 : TruePositive / 비정상-오브젝트없음 : FalseNegative / 정상-오브젝트있음 : FalsePositive / 정상-오브젝트없음 : TrueNegative
                    val_confusion_list = [] ####################################################




                    for idx, acc in enumerate(accs): ###############################################################
                     
                        if 0 == self.valX[0][idx] and acc != 0.:
                            confusion_list.append('TP')
                            val_confusion_list.append('TP')
                    
                        elif 0 == self.valX[0][idx] and acc == 0.:
                            confusion_list.append('FN')
                            val_confusion_list.append('FN')
                   
                        elif 1 == self.valX[0][idx] and acc != 0.:
                            confusion_list.append('FP')
                            val_confusion_list.append('FP')
                  
                        elif 1 == self.valX[0][idx] and acc == 0.:
                            confusion_list.append('TN')
                            val_confusion_list.append('TN')
                 

                    # TP와 TN을 count해서 정확도를 구함#####################################################
                    TP_cnt = val_confusion_list.count('TP')
                    TN_cnt = val_confusion_list.count('TN')
                    val_tot_data_cnt = len(val_confusion_list)
                    val_batch_acc = (TP_cnt + TN_cnt) / val_tot_data_cnt

                    # 진단 정확도 판단을 위해 전체 IoU 의 길이를 받아냅니다. IoU가 0이거나 0에 매우 가까우면 제대로 진단을 하지 못했다고 판단 할 수 있습니다.
                    # 따라서 IoU가 기준치 이상인 값들만 추려내면 질병이 존재한다고 진단 할 수 있습니다.
                    # before_filtered_length = len(ious)

                    # 전체 평균 IoU를 계산합니다.
                    unfiltered_iou = np.mean(ious)

                    # IoU가 0.02 이상, 즉 일정이상 예측해낸 IoU 값들만 모아서 진단 정확도와 질병으로 판단 했을 때의 IoU 값을 따로 계산합니다.
                    iou_list = []

                    for iou in ious:
                        if iou > 0.01:
                            iou_list.append(iou)

                    if len(iou_list) == 0:
                        mean_iou = 0
                    else:
                        mean_iou = np.mean(iou_list)

                    # 배치별 IoU값과 정확도를 전체 IoU값과 정확도에 더합니다. 에폭이 종료되면 평균 IoU와 평균 정확도로 환산합니다.
                    total_val_acc += val_batch_acc ################################
                    total_val_iou += mean_iou
                    total_val_unfiltered_iou += unfiltered_iou

                # 모델을 저장할 경로를 확인하고 없으면 만들어줍니다.
                if os.path.exists(self.model_path + '/' + str(epoch + 1)) is False:
                    os.makedirs(self.model_path + '/' + str(epoch + 1))
                # 모델 저장을 위한 절대경로입니다. '파일명'.ckpt로 저장합니다.
                save_path = self.model_path + '/' + str(epoch + 1) + '/Unet.ckpt'
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
                Valdation_Accuracy = total_val_acc / val_step

                print('Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % self.epoch_num,
                      'Loss =', '{:.4f}  '.format(Loss),
                      'Valdation IoU:{:.4f}   '.format(Valdation_IoU),
                      'Valdation Unfiltered IoU:{:.4f}   '.format(Valdation_Unfiltered_IoU),
                      'Valdation Accuracy:{:.4f}   '.format(Valdation_Accuracy),
                      'Training time: {:.2f}  '.format(training_time))

                result_dict = {self.p_eval.acc: Valdation_Accuracy, self.p_eval.mean_iou: Valdation_IoU,
                               self.p_eval.tot_iou: Valdation_Unfiltered_IoU, self.p_eval.loss: Loss}

                # TB
                summary = sess.run(self.merged_summary, feed_dict=result_dict)
                self.writer.add_summary(summary, global_step=epoch)

                acc_list.append(Valdation_Accuracy)
                mean_iou_list.append(Valdation_IoU)
                unfiltered_iou_list.append(Valdation_Unfiltered_IoU)
                loss_list.append(Loss)

                total_TP = confusion_list.count('TP')
                total_FN = confusion_list.count('FN')
                total_FP = confusion_list.count('FP')
                total_TN = confusion_list.count('TN')

                if not os.path.exists(cfg.LOG_DATA_PATH):
                    os.mkdir(cfg.LOG_DATA_PATH)

                with open(
                        cfg.LOG_DATA_PATH + 'epoch_' + str(
                                epoch + 1) + ".txt", "w", newline='\n') as f:
                    f.write('|  * ACC :{:.4f}'.format(np.max(acc_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * MEAN IOU :{:.4f}'.format(np.max(mean_iou_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * TOTAL IOU :{:.4f}'.format(np.max(unfiltered_iou_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * MIN LOSS :{:.4f}'.format(np.min(loss_list)) + "\n")
                    f.write("|" + "\n")
                    f.write('|  * VALIDATION CONFUSION MATRIX' + "\n")
                    f.write("|" + "\n")
                    f.write('|                     |  Predict True  |  Predict False' + "\n")
                    f.write('|   ------------------+----------------+--------------------' + "\n")
                    f.write('|   Groudtruth True   |      {0:^4}      |      {1}'.format(total_TP, total_FN) + "\n")
                    f.write("|   Groudtruth False  |      {0:^4}      |      {1}".format(total_FP, total_TN) + "\n")
                    f.write("|" + "\n")
                    f.write(
                        '|  * VALIDATION SENSITIVITY : {:.4f}'.format(total_TP / (total_TP + total_FN + 1e-6)) + "\n")
                    f.write("|" + "\n")
                    f.write(
                        '|  * VALIDATION SPECIFICITY : {:.4f}'.format(total_TN / (total_FP + total_TN + 1e-6)) + "\n")
                    f.write("|" + "\n")
                    f.write("=================================================================")

            print("")
            print("TRAINING COMPLETE")


if __name__ == '__main__':
    unet_trainer = Trainer(training_data_path=[cfg.TRAIN_DATA_PATH],
                           validation_data_path=[cfg.VAL_DATA_PATH],
                           pkl_data_path=cfg.PKL_DATA_PATH,
                           model_path='./model',
                           validation_percentage=10,
                           initial_learning_rate=0.005,
                           decay_step=2500,
                           decay_rate=0.9,
                           epoch=400,
                           img_size=256,
                           n_class=1,
                           batch_size=28,
                           upsampling_mode='transpose',  # resize, transpose
                           normalization_mode='batch',  # batch, group
                           model_root_channel=32,
                           activation='prelu',  # lrelu, prelu, selu, elu, relu, None
                           saving_epoch=2,
                           loss='dice_sum',  # dice, focal, huber, cross_entropy, dice_sum
                           groupnormalization_n=None,
                           channel_mode='NHWC',
                           downsampling_option='maxpool')  # linear, neighbor, maxpool
