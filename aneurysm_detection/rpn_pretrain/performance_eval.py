import tensorflow as tf

class performance:
    def __init__(self):
        # 텐서보드 스칼라 서머리용 플레이스홀더
        self.acc = tf.placeholder(tf.float32)
        self.sens = tf.placeholder(tf.float32)
        self.spec = tf.placeholder(tf.float32)
        self.miou = tf.placeholder(tf.float32)
        self.dice = tf.placeholder(tf.float32)
        self.hdorff = tf.placeholder(tf.float32)

        # 텐서 스칼라 값을 텐서보드에 기록합니다.
        tf.summary.scalar('Accuracy', self.acc)
        tf.summary.scalar('Sensitivity', self.sens)
        tf.summary.scalar('Specificity', self.spec)
        tf.summary.scalar('Mean IoU', self.miou)
        tf.summary.scalar('Dice Score', self.dice)
        tf.summary.scalar('Hausdorff Distance', self.hdorff)
