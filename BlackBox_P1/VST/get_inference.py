from mmaction.apis import inference_recognizer, init_recognizer

# Choose to use a config and initialize the recognizer
config = './VST/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
# Setup a checkpoint file to load
checkpoint = './VST/checkpoints/swint.pth'
# Initialize the recognizer
model = init_recognizer(config, checkpoint, device='cuda:0')
label = './VST/tools/data/kinetics/label_map_k400.txt'

def get_logits(video_path):
	video = video_path
	results = inference_recognizer(model, video,label)
	return(results)
