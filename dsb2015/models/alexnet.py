import mxnet as mx

def get_alexnet():
    """ A alexnet style net, takes difference of each frame as input."""
    source = mx.sym.Variable("data")
    source = (source - 128) * (1.0/128)
    # frames = mx.sym.SliceChannel(source, num_outputs=30)
    # # diffs = [frames[i+1] - frames[i] for i in range(29)]
    # # source = mx.sym.Concat(*diffs)
    # source = mx.sym.Concat(*frames)
    net = mx.sym.Convolution(source, kernel=(4, 4), num_filter=32)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    # net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    net = mx.sym.Convolution(net, kernel=(3, 3), num_filter=128)
    net = mx.sym.BatchNorm(net, fix_gamma=True)
    net = mx.sym.Activation(net, act_type="relu")
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(net)
    flatten = mx.symbol.Dropout(flatten)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=2048)
    fc1 = mx.symbol.Dropout(fc1)
    fc1 = mx.sym.Activation(fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=2048)
    fc2 = mx.symbol.Dropout(fc2)
    fc2 = mx.sym.Activation(fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2, num_hidden=600)
    # Name the final layer as softmax so it auto matches the naming of data iterator
    # Otherwise we can also change the provide_data in the data iter
    return mx.symbol.LogisticRegressionOutput(data=fc3, name='softmax')
