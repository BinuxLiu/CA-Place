from models.dinov2_network import DINOv2
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
import torch

def test_demo():
    model = DINOv2(backbone = "dinov2_vitb14")
    # eval mode for inference
    model.cuda().eval()

    # prepare image for the model
    url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    input_resolution = (3, 448, 448)  # MambaVision supports any input resolutions

    transform = create_transform(input_size=input_resolution)

    inputs = transform(image).unsqueeze(0).cuda()

    import time

    # Number of iterations to average throughput calculation
    num_iterations = 3000

    # Warm-up pass (important for accurate timing in CUDA)
    for _ in range(300):
        _, _ = model(inputs)

    # Timing the inference
    start_time = time.time()
    for _ in range(num_iterations):
        outputs = model(inputs)
    torch.cuda.synchronize()  # Synchronize CUDA to ensure accurate timing

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate throughput
    throughput = num_iterations / elapsed_time  # frames per second (FPS)
    print(f"Throughput: {throughput:.2f} FPS")
    

if __name__=="__main__":
    test_demo()