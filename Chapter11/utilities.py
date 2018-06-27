
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

def genBatch( X, y, batchSize ):
    """Generator of batches."""

    inds = np.random.permutation( len(X) )

    for start in range(0, len(X) - 1, batchSize):

        yield X[ inds[start : start + batchSize] ], y[ inds[start : start + batchSize] ]


def mnistClassifier( X, y, training, nOut, nl = 1, nh = 100, alpha = 0.01, b1 = 0.9, b2 = 0.999 ):

    if ( nl < 1 ):
        print( "You need at least one hidden layer." )
        return

    if ( nh < 1 ):
        print( "you need at least one neuron." )
        return

    with tf.name_scope( "dnn" ):

        l1 = tf.layers.dense( X, nh, name = "hidden1",
                              kernel_initializer = tf.keras.initializers.he_normal() )
        bn1 = tf.layers.batch_normalization( l1, training = training, momentum = b1 )
        bnAct1 = tf.nn.elu( bn1 )

        layers = [ bnAct1 ]

        for i in range(2, nl + 1):

            li = tf.layers.dense( layers[-1], nh, name = "hidden" + str(i),
                                            kernel_initializer = tf.keras.initializers.he_normal() )
            bni = tf.layers.batch_normalization( li, training = training, momentum = b1 )
            bnActi = tf.nn.elu( bni )
            layers.append( bnActi )

        ln = tf.layers.dense( layers[-1], nOut, name = "output",
                                  kernel_initializer = tf.keras.initializers.he_normal() )
        logits = tf.layers.batch_normalization( ln, training = training, momentum = b1 )

    with tf.name_scope("loss"):
        crossEnt = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = y, logits = logits)
        loss = tf.reduce_mean( crossEnt, name = "loss" )

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )

    with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer( learning_rate = alpha, beta1 = b1, beta2 = b2 )
        training = opt.minimize( loss )
        lossSummary = tf.summary.scalar("crossEntropy", loss)

    with tf.name_scope("utility"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return loss, training, accuracy, lossSummary, init, saver

def trainModel( trainX, trainY, valX, valY, params, saveModel = False ):

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (None, 28*28), name = "X")
    y = tf.placeholder(tf.int32, shape = (None), name = "y")
    trn = tf.placeholder_with_default( False, shape = (), name = "trn" )

    loss, training, accuracy, lossSummary, init, saver = mnistClassifier( X, y, trn, **params )

    nEpochs = 5000
    batchSize = 200

    loVal = 10000
    patience = 0

    step = 0

    with tf.Session() as sess:

        init.run()

        tls = [ loss.eval( feed_dict = { X : trainX, y : trainY } ) ]
        vls = [ loss.eval( feed_dict = { X : valX, y : valY } ) ]

        for epoch in range(nEpochs):
            for batchX, batchY in genBatch( trainX, trainY, batchSize ):

                sess.run( training, feed_dict = { X : batchX, y : batchY } )
                step += 1

                if ( step % 50 == 0 ):
                    valLoss = loss.eval( feed_dict = { X : valX, y : valY, trn : True } )

                    if ( valLoss < loVal ):
                        loVal = valLoss

                        if (saveModel):
                            saver.save( sess, "./mnist-best.ckpt" )

            trainLoss = loss.eval( feed_dict = { X : trainX, y : trainY } )
            valLoss   = loss.eval( feed_dict = { X : valX, y : valY } )

            tls.append( trainLoss )
            vls.append( valLoss )

            if ( valLoss < loVal ):
                loVal = valLoss
                patience = 0

            else:
                patience += 1

            if ( patience >= 50):
                break

    return (loVal, tls, vls)

def crossValidation( trainX, trainY, k, params ):

    res = []
    kf = KFold( n_splits = k )

    for indTr, indVl in kf.split( trainX, trainY ):
        valLoss , tls, vls = trainModel( trainX[indTr], trainY[indTr],
                                         trainX[indVl], trainY[indVl], params )

        res.append(valLoss)

    return np.mean( res )

def hyperparameterSearch( trainX, trainY, paramsList, k ):

    bestLoss = 100000
    modelID = 0
    bestParams = None

    for params in paramsList:

        t1 = time.time()
        valLoss = crossValidation( trainX, trainY, k, params )
        t2 = time.time()

        if ( valLoss < bestLoss ):
            bestLoss = valLoss
            bestParams = params

        print("Done {0} of {1} in {2}s".format(modelID + 1, len(paramsList), t2 - t1) )
        modelID += 1

    trX, valX, trY, valY = train_test_split( trainX, trainY, test_size = 10000,
                                                   random_state = 123 )

    _, trHist, valHist = trainModel( trX, trY, valX, valY, bestParams, saveModel = True )

    return trHist, valHist, bestParams

