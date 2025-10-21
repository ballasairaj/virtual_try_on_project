from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import warnings
import time

warnings.filterwarnings("ignore")
def load_model_a():
    # Start total execution timer
    total_start_time = time.time()

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")

    # Load and move model to GPU
    model_start_time = time.time()
    model = create_model("Unet_2020-10-30")
    model = model.to(device)
    model.eval()
    model_load_time = time.time() - model_start_time
    print(f"Model loading time: {model_load_time:.2f} seconds")
    return model, device, total_start_time

def process_image_using_a_model(model, device, total_start_time):
    # Load image
    image_load_start_time = time.time()
    image = load_rgb("./static/cloth_web.jpg")
    image_load_time = time.time() - image_load_start_time
    print(f"Image loading time: {image_load_time:.2f} seconds")

    # Define transformation
    transform = albu.Compose([albu.Normalize(p=1)], p=1)

    # Pad image
    pad_start_time = time.time()
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    pad_time = time.time() - pad_start_time
    print(f"Padding time: {pad_time:.2f} seconds")

    # Transform and prepare tensor for GPU
    transform_start_time = time.time()
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0).to(device)
    transform_time = time.time() - transform_start_time
    print(f"Transform and tensor preparation time: {transform_time:.2f} seconds")

    # Perform inference on GPU
    inference_start_time = time.time()
    with torch.no_grad():
        prediction = model(x)[0][0]
    inference_time = time.time() - inference_start_time
    print(f"Inference time: {inference_time:.2f} seconds")

    # Move prediction back to CPU and process
    postprocess_start_time = time.time()
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    postprocess_time = time.time() - postprocess_start_time
    print(f"Post-processing (mask creation) time: {postprocess_time:.2f} seconds")

    # Initialize output images
    img = np.full((1024, 768, 3), 255, dtype=np.uint8)
    seg_img = np.full((1024, 768), 0, dtype=np.uint8)

    # Load and process input image
    image_process_start_time = time.time()
    b = cv2.imread("./static/cloth_web.jpg")
    b_img = mask * 255

    # Resize if needed
    if b.shape[1] <= 600 and b.shape[0] <= 500:
        b = cv2.resize(b, (int(b.shape[1] * 1.2), int(b.shape[0] * 1.2)))
        b_img = cv2.resize(b_img, (int(b_img.shape[1] * 1.2), int(b_img.shape[0] * 1.2)))

    # Get shape and place images
    shape = b_img.shape
    img[int((1024 - shape[0]) / 2): 1024 - int((1024 - shape[0]) / 2), int((768 - shape[1]) / 2): 768 - int((768 - shape[1]) / 2)] = b
    seg_img[int((1024 - shape[0]) / 2): 1024 - int((1024 - shape[0]) / 2), int((768 - shape[1]) / 2): 768 - int((768 - shape[1]) / 2)] = b_img
    image_process_time = time.time() - image_process_start_time
    print(f"Image processing and resizing time: {image_process_time:.2f} seconds")

    # Save output images
    save_start_time = time.time()
    cv2.imwrite("./HR_VITON_main/test/test/cloth/00001_00.jpg", img)
    cv2.imwrite("./HR_VITON_main/test/test/cloth-mask/00001_00.jpg", seg_img)
    save_time = time.time() - save_start_time
    print(f"Image saving time: {save_time:.2f} seconds")

    # Total execution time
    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")

