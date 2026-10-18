from pathlib import Path
import argparse,json,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'projects/rtdetrv4_uav'))
from bootstrap import write_json
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from run_query_difference import build
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
a=p.parse_args()
output=Path(a.output).resolve()
if root!=output and root not in output.parents:raise ValueError('Output must be inside this extracted F-drive project')
model,criterion,post=build(a.device)
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model.load_state_dict(checkpoint['model'],strict=True);model.eval()
with Image.open(a.image) as f:image=f.convert('RGB')
w,h=image.size
x=pil_to_tensor(image.resize((640,640),Image.Resampling.BILINEAR)).float().div(255).unsqueeze(0).to(a.device)
with torch.inference_mode():pred=post(model(x),torch.tensor([[w,h]],device=a.device))[0]
write_json(output,{'image':str(Path(a.image).resolve()),'boxes_xyxy':pred['boxes'].cpu().tolist(),'scores':pred['scores'].cpu().tolist(),'categories_1_to_10':(pred['labels']+1).cpu().tolist()})
