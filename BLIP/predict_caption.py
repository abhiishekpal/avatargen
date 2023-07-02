from models.blip import blip_decoder
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_demo_image(image_size,device):
    raw_image = Image.open("C:/Users/91973/Documents/Kaggle/val2017/val2017/000000000785.jpg").convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 384
image = load_demo_image(image_size=image_size, device=device)

model_path = "C:/Users/91973/Documents/Kaggle/BLIP-main/BLIP-main/checkpoints/model__base_caption.pth"
    
model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)
import tqdm
with torch.no_grad():
    # beam search
    for _ in tqdm.tqdm(range(100)):
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        print('caption: '+caption[0])