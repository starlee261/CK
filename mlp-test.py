#-*- coding: utf-8 -*-  


from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import cPickle
import random

import theano
import theano.tensor as T


#参数说明：  
#input，输入的一个batch，假设一个batch有n个样本(n_example)，则input大小就是(n_example,n_in)  
#n_in,每一个样本的大小，CK+每个样本是一张64*64的图片，故n_in=4096  
#n_out,输出的类别数，MNIST有0～7共8个类别，n_out=8   
class LogisticRegression(object):  
    def __init__(self, input, n_in, n_out):  
  
#W大小是n_in行n_out列，b为n_out维向量。即：每个输出对应W的一列以及b的一个元素。WX+b    
#W和b都定义为theano.shared类型，这个是为了程序能在GPU上跑。  
        self.W = theano.shared(  
            value=numpy.zeros(  
                (n_in, n_out),  
                dtype=theano.config.floatX  
            ),  
            name='W',  
            borrow=True  
        )  
  
        self.b = theano.shared(  
            value=numpy.zeros(  
                (n_out,),  
                dtype=theano.config.floatX  
            ),  
            name='b',  
            borrow=True  
        )  
  
#input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，  
#再作为T.nnet.softmax的输入，得到p_y_given_x  
#故p_y_given_x每一行代表每一个样本被估计为各类别的概率      
#PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，  
#然后(n_example,n_out)矩阵的每一行都加b  
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)  
  
#argmax返回最大值下标，因为本例数据集是MNIST，下标刚好就是类别。axis=1表示按行操作。  
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)  
  
#params，模型的参数       
        self.params = [self.W, self.b]  

# keep track of model input
        self.input = input
  
#代价函数NLL  
#因为我们是MSGD，每次训练一个batch，一个batch有n_example个样本，则y大小是(n_example,),  
#y.shape[0]得出行数即样本数，将T.log(self.p_y_given_x)简记为LP，  
#则LP[T.arange(y.shape[0]),y]得到[LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,LP[n-1,y[n-1]]]  
#最后求均值mean，也就是说，minibatch的SGD，是计算出batch里所有样本的NLL的平均值，作为它的cost  
    def negative_log_likelihood(self, y):    
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])  
  
#batch的误差率  
    def errors(self, y):  
        # 首先检查y与y_pred的维度是否一样，即是否含有相等的样本数  
        if y.ndim != self.y_pred.ndim:  
            raise TypeError(  
                'y should have the same shape as self.y_pred',  
                ('y', y.type, 'y_pred', self.y_pred.type)  
            )  
        # 再检查是不是int类型，是的话计算T.neq(self.y_pred, y)的均值，作为误差率  
        #举个例子，假如self.y_pred=[3,2,3,2,3,2],而实际上y=[3,4,3,4,3,4]  
        #则T.neq(self.y_pred, y)=[0,1,0,1,0,1],1表示不等，0表示相等  
        #故T.mean(T.neq(self.y_pred, y))=T.mean([0,1,0,1,0,1])=0.5，即错误率50%  
        if y.dtype.startswith('int'):  
            return T.mean(T.neq(self.y_pred, y))  
        else:  
            raise NotImplementedError() 



def read_data(dataset):
   
    read_file=open(dataset,'rb')  
    faces=cPickle.load(read_file)    
    label=cPickle.load(read_file)    
    read_file.close()   

    train_data=numpy.empty((2072,4096))  
    train_label=numpy.empty(2072)  
    valid_data=numpy.empty((444,4096))  
    valid_label=numpy.empty(444)  
    test_data=numpy.empty((444,4096))  
    test_label=numpy.empty(444)  
      
    li=range(2960)
    random.shuffle(li)

    for i in range(2960):
        if (i<2072):
            train_data[i]=faces[li[i]]
            train_label[i]=label[li[i]]  
        elif (i>=2072 and i<2516):
            valid_data[i-2072]=faces[li[i]]  
            valid_label[i-2072]=label[li[i]]
        else:  
            test_data[i-2516]=faces[li[i]]  
            test_label[i-2516]=label[li[i]] 

    train_label=train_label.astype(numpy.int)
    valid_label=valid_label.astype(numpy.int)
    test_label=test_label.astype(numpy.int)


#将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中  
#GPU里数据类型只能是float。而data_y是类别，所以最后又转换为int返回  
    def shared_dataset(data_x,data_y, borrow=True):  
        shared_x = theano.shared(numpy.asarray(data_x,  
                                               dtype=theano.config.floatX),  
                                 borrow=borrow)  
        shared_y = theano.shared(numpy.asarray(data_y,  
                                               dtype=theano.config.floatX),  
                                 borrow=borrow)  
        return shared_x, T.cast(shared_y, 'int32')  
  
  
    test_set_x, test_set_y = shared_dataset(test_data,test_label)  
    valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)  
    train_set_x, train_set_y = shared_dataset(train_data,train_label)  
  
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),  
            (test_set_x, test_set_y)]  
    return rval 


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.001, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='D:\\CK\\data.pkl', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = read_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=64 * 64,
        n_hidden=n_hidden,
        n_out=8
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    test_mlp()