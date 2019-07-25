## mnist手写数字识别
- 深度学习和TensorFlow入门

### Errors
1. Error-1
- ValueError: Variable conv1/weight/ExponentialMovingAverage/ does not exist, or was not created with tf.get_variable(). 
Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
- 原因:  
因为将控制变量重用的参数reuse设为True, 所以get_variable函数不会创建新变量, 而是复用已有变量. 
若不存在与变量名对应的变量, 就会报错. 而ema需要用get_variable创建新的变量.
- Solve methods:  
    - [x] 在reuse设为True之前, 定义ema及其操作.  
    - [X] 退出for循环后, 将reuse恢复其初始值.  
    - [X] 定义一个与当前variable_scope同名的variable_scope, reuse设为True.

