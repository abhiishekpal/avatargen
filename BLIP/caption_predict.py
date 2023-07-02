import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder
import configs.constants as C

class BlipPredict:

    def __init__(self) -> None:

        self.image_size = 384
        self.num_beams = 3
        self.max_length = 20
        self.min_length = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = C.BLIP_MODEL_PATH
        self.model = blip_decoder(pretrained=model_path, image_size=self.image_size, vit='base')
        self.model.eval()
        self.model = self.model.to(self.device)

    def load_demo_image(self, path):

        raw_image = Image.open(path).convert('RGB')   
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(self.device)   
        return image

    def predict(self, path):

        image = self.load_demo_image(path)
        with torch.no_grad():
            caption = self.model.generate(image, sample=False, num_beams=self.num_beams, max_length=self.max_length, min_length=self.min_length)  
        return caption