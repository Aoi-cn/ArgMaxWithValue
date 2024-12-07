# 进度汇报

* 这个项目目前由ArgMaxWithValue.json文件，通过`${ASCEND_HOME_PATH}/python/site-packages/bin/msopgen gen -i ArgMaxWithValue.json -c ai_core-Ascend910b -lan cpp -out .`命令生成（不一定准确）

* 目前比较奇怪的事情有：
    * 性能要求中要求适配uint8，但我看别人写的都是int8，到时候如果报错可能需要重新生成算子工程（我这里写的是uint8）