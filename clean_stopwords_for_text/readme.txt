此部分为去除停止词，由于文件过大，则分开处理
执行文件之前需要创建文件夹cleared_dataset
并且在此文件夹下创建文件label名
因为一共10个分类，所以手动创建也不麻烦，主要是使用并行处理，加快预处理速度