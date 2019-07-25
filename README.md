## mnist手写数字识别
- 深度学习和TensorFlow入门
- train_dnn.py
    - 两层全连接层
- train_cnn.py/train_eval_cnn.py/train_eval_cnn_monitor.py
    - 卷积神经网络：两层block（卷积+池化）+ 两层全连接
- train_eval_cnn_multi_gpu.py
    - 卷积神经网络，多GPU训练
- train_eval_cnn_multi_process.py
    - 使用multiprocessing多进程并行训练卷积神经网络。每个进程的训练模型有不同的超参数。进程间没有通信。可以用于模型调参。
- train_eval_cnn_multi_process_communication.py
    - 使用multiprocessing多进程并行训练卷积神经网络。使用queue, pipe, manager作进程间通信。代码没有实际用处，只是练习进程间通信。
- Errors
    1. Error-1
        - ValueError: Variable conv1/weight/ExponentialMovingAverage/ does not exist, or was not created with tf.get_variable(). 
        Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
        -  原因: 
        因为将控制变量重用的参数reuse设为True, 所以get_variable函数不会创建新变量, 而是复用已有变量. 
        若不存在与变量名对应的变量, 就会报错. 而ema需要用get_variable创建新的变量.
        - Solve methods:  
            - [x] 在reuse设为True之前, 定义ema及其操作.  
            - [X] 退出for循环后, 将reuse恢复其初始值.  
            - [X] 定义一个与当前variable_scope同名的variable_scope, reuse设为True.

