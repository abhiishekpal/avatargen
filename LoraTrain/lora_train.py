import subprocess
import shutil


class LoraTrain:

    def __init__(self, base_path) -> None:

        self.base_path = base_path
        self.pretrained_model = r"runwayml/stable-diffusion-v1-5"
        
    def train(self, name_path):

        self.train_data_dir = r"{}\{}\image".format(self.base_path, name_path)
        self.output_dir = r"{}\{}\model".format(self.base_path, name_path)
        self.log_dir = r"{}\{}\log".format(self.base_path, name_path)
        self.output_name = name_path.split('_')[0]

        print(self.train_data_dir)
        print(self.output_dir)
        print(self.log_dir)

        run_cmd = f'python LoraTrain\\train_network.py --pretrained_model_name_or_path="{self.pretrained_model}" --train_data_dir="{self.train_data_dir}" --resolution=512,512 --output_dir="{self.output_dir}" --logging_dir="{self.log_dir}" --network_alpha="128" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=5e-5 --unet_lr=0.0001 --network_dim=128 --output_name={self.output_name} --lr_scheduler_num_cycles="1" --learning_rate="0.0001" --lr_scheduler="constant" --train_batch_size="1" --max_train_steps="1904" --save_every_n_epochs="1" --mixed_precision="no" --save_precision="float" --seed="1234" --caption_extension=".txt" --cache_latents --max_data_loader_n_workers="1" --clip_skip=2 --bucket_reso_steps=64 --mem_eff_attn --gradient_checkpointing --xformers --use_8bit_adam --bucket_no_upscale'
        run_cmd = r"{}".format(run_cmd)
        print(run_cmd)
        subprocess.run(run_cmd)

        src = self.output_dir + f'/{self.output_name}.safetensor'
        dst = '../StableDiffusion/models/Lora'  
        shutil.copyfile(src, dst)


