
import time
import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, KFold


def genBatch( X, y, batchSize ):
    """Generator of batches."""

    inds = np.random.permutation( len(X) )

    for start in range(0, len(X) - 1, batchSize):

        yield X[ inds[start : start + batchSize] ], y[ inds[start : start + batchSize] ]

def mnistCNN( X, y, trn, alpha = 0.01, momentum = 0.9, bnm = 0.9 ):

    with tf.name_scope( "cnn" ):
        #LeNet: 6, 16, 120, FC: 84

        pad = tf.image.resize_image_with_crop_or_pad( X, 32, 32 )

        conv1 = tf.layers.conv2d( pad, filters = 4, kernel_size = 5, strides = 1,
                                  padding = "SAME", name = "conv1" )
        bn1 = tf.layers.batch_normalization( conv1, training = trn, momentum = bnm )
        bnAct1 = tf.nn.elu( bn1 )

        pool1 = tf.layers.average_pooling2d( bnAct1, pool_size = 2,
                                             strides = 2, name = "pool1" )

        conv2 = tf.layers.conv2d( pool1, filters = 8, kernel_size = 5, strides = 1,
                                  padding = "SAME", name = "conv2" )
        bn2 = tf.layers.batch_normalization( conv2, training = trn, momentum = bnm )
        bnAct2 = tf.nn.elu( bn2 )

        pool2 = tf.layers.average_pooling2d( bnAct2, pool_size = 2,
                                             strides = 2, name = "pool2" )

        conv3 = tf.layers.conv2d( pool2, filters = 8, kernel_size = 1, strides = 1,
                                  padding = "SAME", name = "conv3" )
        bn3 = tf.layers.batch_normalization( conv3, training = trn, momentum = bnm )
        bnAct3 = tf.nn.elu( bn3 )

        flat = tf.layers.flatten( bnAct3 )

        fc1 = tf.layers.dense( flat, 10, name = "fc1", activation = tf.nn.elu,
                               kernel_initializer = tf.keras.initializers.he_normal() )

        logits = tf.layers.dense( fc1, 10, name = "output" )

    with tf.name_scope("loss"):
        crossEnt = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = y, logits = logits)
        loss = tf.reduce_mean( crossEnt, name = "loss" )

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean( tf.cast(correct, tf.float32) )

    with tf.name_scope("train"):
        opt = tf.train.MomentumOptimizer( learning_rate = alpha,
                                          momentum = momentum,
                                          use_nesterov = True )
        training = opt.minimize( loss )
        lossSummary = tf.summary.scalar("crossEntropy", loss)

    with tf.name_scope("utility"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return loss, training, accuracy, lossSummary, init, saver

def trainModel( trainX, trainY, valX, valY, params, saveModel = False ):

    parameters = params[ "params" ]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape = (None, 28, 28, 1), name = "X")
    y = tf.placeholder(tf.int32, shape = (None), name = "y")
    trn = tf.placeholder_with_default( False, shape = (), name = "trn" )

    loss, training, accuracy, lossSummary, init, saver = mnistCNN( X, y, trn, **parameters )

    nEpochs = 5000
    batchSize = params[ "batchSize" ]

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
                            saver.save( sess, "./best/mnist-best.ckpt" )

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

        if (saveModel):
            print()
            print("***")
            print("Final model validation accuracy:", accuracy.eval(feed_dict = { X : valX, y : valY }) )
            print("***")
            print()

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

        print("Done {0} of {1} in {2}s".format(modelID + 1, len(paramsList), t2 - t1),
              "Validation loss:", valLoss )
        modelID += 1

    trX, valX, trY, valY = train_test_split( trainX, trainY, test_size = 10000,
                                                   random_state = 123 )

    loVal, trHist, valHist = trainModel( trX, trY, valX, valY, bestParams, saveModel = True )

    return ( loVal, trHist, valHist, bestParams )

