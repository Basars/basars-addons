import tensorflow as tf

from tensorflow.keras.optimizers.schedules import CosineDecayRestarts


class InitialCosineDecayRestarts(CosineDecayRestarts):

    def __init__(self,
                 initial_epoch,
                 initial_learning_rate,
                 first_decay_steps,
                 t_mul=2.0,
                 m_mul=1.0,
                 alpha=0.0,
                 name=None):
        super(InitialCosineDecayRestarts, self).__init__(initial_learning_rate,
                                                         first_decay_steps,
                                                         t_mul, m_mul,
                                                         alpha, name)

        self.initial_epoch = initial_epoch

    def __call__(self, step):
        step = step + self.initial_epoch

        return super(InitialCosineDecayRestarts, self).__call__(step)


class CosineDecayWarmupRestarts(CosineDecayRestarts):

    def __init__(self,
                 first_cycle_steps,
                 initial_learning_rate,
                 first_decay_steps,
                 t_mul=2.0,
                 m_mul=1.0,
                 alpha=0.0,
                 name=None):
        super(CosineDecayWarmupRestarts, self).__init__(initial_learning_rate,
                                                        first_decay_steps,
                                                        t_mul, m_mul,
                                                        alpha, name)
        self.first_cycle_steps = first_cycle_steps

    def __call__(self, step):
        stage1 = tf.cast(step > self.first_cycle_steps, tf.float32)
        stage2 = tf.cast(step <= self.first_cycle_steps, tf.float32)
        annealing = super(CosineDecayWarmupRestarts, self).__call__(step - self.first_cycle_steps) * stage1
        warmup = self.initial_learning_rate * step * (self.first_cycle_steps ** -1) * stage2
        return tf.add(annealing, warmup)
