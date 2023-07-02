import sys
sys.path.append('./BLIP')
sys.path.append('./LoraTrain')
sys.path.append('./library')
sys.path.append('./')

from train import Trainer


obj = Trainer()
path = r'C:\Users\91973\Desktop\Myself\test'
obj.train(path)
