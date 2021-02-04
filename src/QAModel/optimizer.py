import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # def get_config(self):
    #     config = {
    #       'd_model': self.d_model,
    #       'warmup_steps': self.warmup_steps,
    #
    #     }
    #     return config



class Optimizer():
    def __init__(self):
        self.MAX_LENGTH = 40
        self.D_MODEL = 128

    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def AdamOptimizer(self):
        learning_rate = CustomSchedule(self.D_MODEL)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        return optimizer, self.loss_function, self.accuracy


    def accuracy(self, y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)








