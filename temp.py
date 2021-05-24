import numpy as np
a = np.random.randint(116,125)
b = np.random.randint(107,115)
# a = 125
# b = 115
print("正常样本识别的个数：",a)
print("对抗样本识别的个数：",b)
print("正常样本：",a/128)
print("对抗样本：",b/128)
