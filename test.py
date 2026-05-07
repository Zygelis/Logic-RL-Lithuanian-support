import torch
import time
 
val = torch.cuda.is_available()
 
f = open("gpu.txt", "a")
ts = time.time()
f.write("GPU is loaded {} at {}\n".format(val, ts))
print("GPU is loaded {} at {}".format(val, ts))
f.close()