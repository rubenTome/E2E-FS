import os

N = 10

for i in range(N):
    os.system("python3 example.py float16 " + str(i))
    os.system("python3 example.py float32 " + str(i))