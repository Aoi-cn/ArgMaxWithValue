# 进度汇报

* 这个项目目前由ArgMaxWithValue.json文件，通过`${ASCEND_HOME_PATH}/python/site-packages/bin/msopgen gen -i ArgMaxWithValue.json -c ai_core-Ascend910b -lan cpp -out .`命令生成（不一定准确）

* 目前比较奇怪的事情有：
    * 性能要求中要求适配uint8，但我看别人写的都是int8，到时候如果报错可能需要重新生成算子工程（我这里写的是uint8）


* 可以参考的项目有sample里的`归档\samples\operator_contrib\UnalignAddCustomSample`这个，这是一个非对其add的实现

* 测试数据格式：
```
# data1
input_x = np.random.uniform(-10, 10, [64, 64]).astype(np.float16)
dimension = 0
keep_dims = False

# data2
input_x = np.random.uniform(-1000, 1000, [32, 64]).astype(np.float32)
dimension = 1
keep_dims = False

# data3
input_x = np.random.uniform(-100, 100, [3, 1280, 640]).astype(np.int32)
dimension = 2
keep_dims = True

# data4
input_x = np.random.uniform(0, 127, [13, 171, 351]).astype(np.uint8)
dimension = 0
keep_dims = False
```