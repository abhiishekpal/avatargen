import os
import shutil
import random
import tqdm

from BLIP.caption_predict import BlipPredict
from LoraTrain.lora_train import LoraTrain

import configs.constants as C

class Trainer:

    def __init__(self) -> None:
        
        self.blip = BlipPredict()
        self.data_path = C.DATA_DIR
        self.lora = LoraTrain(base_path=self.data_path)


    def process_data(self, path, your_name):

        new_folder = os.path.join(self.data_path, your_name+'_'+'PERSON')
        while os.path.exists(new_folder):
            your_name = ''.join([random.choice('xvwzy') for _ in range(4)])
            your_name = your_name.upper()
            new_folder = os.path.join(self.data_path, your_name+'_'+'PERSON')
        
        total_images = len([it for it in os.listdir(path)])
        steps = max(100, int(1500/total_images))

        os.mkdir(new_folder)
        os.mkdir(os.path.join(new_folder, 'image'))
        os.mkdir(os.path.join(new_folder, 'image', f'{steps}_{your_name.lower()}'))
        os.mkdir(os.path.join(new_folder, 'log'))
        os.mkdir(os.path.join(new_folder, 'model'))
         
        
        dst_path = os.path.join(new_folder, 'image', f'{steps}_{your_name.lower()}')

        for fl in tqdm.tqdm(os.listdir(path), desc='Images Processed'):
            path_fl = os.path.join(path, fl)
            file_name = os.path.basename(path_fl)
            caption = self.blip.predict(path_fl)[0]
            caption_file_name = os.path.join(dst_path, file_name+'.txt')
            caption = your_name + " " + caption
            with open(caption_file_name, 'w') as fp:
                fp.write(caption+'\n')
            src = path_fl
            shutil.copy2(src, dst_path)

        return your_name + '_PERSON'

    def train(self, path, your_name = "OHWX"):
        
        success = self.process_data(path, your_name)

        if not success:
            return
        
        print('Starting Train...')
        self.lora.train(success)

