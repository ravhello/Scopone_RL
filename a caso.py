import tensorflow as tf
print(tf.sysconfig.get_build_info()["cuda_version"])

print("ora cuda per pytorch")
import torch
print(torch.version.cuda)