从普通的模型保存，伪节点量化模型保存（大小几乎没变，只是用来判断哪些节点可以量化，并且记录min，max）,固化，最后将pb文件转化成tflite文件：

normal_train.py -> quant_aware_train.py -> freeze.py -> convert_to_lite.py 
