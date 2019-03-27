import os
import torchaudio

for filename in os.listdir(os.getcwd()):
    try:
        a = torchaudio.load(filename)
    except:
        print(filename)

print("all good")