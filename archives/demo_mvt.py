from models.mamba_vision import mamba_vision_T
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
import torch

model = mamba_vision_T(model_path = "autodl-tmp/MambaVision/weights/mambavision_tiny_1k.pth.tar")
# eval mode for inference
model.cpu().eval()

# prepare image for the model
url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
image = Image.open(requests.get(url, stream=True).raw)
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions

transform = create_transform(input_size=input_resolution)

inputs = transform(image).unsqueeze(0).cpu()
# model inference
x = model.patch_embed(inputs)
for level in model.levels:
    x = level(x)
    print(x.shape)
x = self.norm(x)
x = self.avgpool(x)
x = torch.flatten(x, 1)
print(x.shape)


# import time

# # Number of iterations to average throughput calculation
# num_iterations = 3000

# # Warm-up pass (important for accurate timing in CUDA)
# for _ in range(300):
#     _ = model.forward_features(inputs)

# # Timing the inference
# start_time = time.time()
# for _ in range(num_iterations):
#     outputs = model.forward_features(inputs)
# torch.cuda.synchronize()  # Synchronize CUDA to ensure accurate timing

# end_time = time.time()
# elapsed_time = end_time - start_time

# # Calculate throughput
# throughput = num_iterations / elapsed_time  # frames per second (FPS)
# print(f"Throughput: {throughput:.2f} FPS")
