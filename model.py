import tensorflow as tf


class model(tf.keras.Model):
    def __init__(self, stateSize, actionSize, optimizer=None, loss = None):
        super(model,self).__init__()

        self.dense1 = tf.keras.layers.Dense(16,input_shape=(stateSize,),activation='relu')

        self.recur = tf.keras.layers.LSTM(128)

        self.dense3 = tf.keras.layers.Dense(actionSize*actionSize,activation='sigmoid')

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001) if optimizer is None else optimizer

        self.loss = tf.keras.losses.MeanSquaredError() if loss is None else loss


    def call(self,x):
        x = self.dense1(x)
        x = self.recur(x)
        return self.dense3(x)

    @tf.function
    def trainStep(self,x,y):
        with tf.GradientTape() as tape:

            pred = self.call(x)
            currLoss = self.loss(y,pred)
        gradients = tape.gradient(currLoss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))

