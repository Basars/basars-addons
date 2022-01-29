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
