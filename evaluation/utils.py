from PIL import Image
import io
from io import BytesIO
import base64
import ast
import json
import traceback
import math
from pathlib import Path
import os


def pil_to_base64(image):
    # Create a BytesIO object
    buffered = io.BytesIO()
    
    # Save the image into the BytesIO object in PNG format
    image.save(buffered, format="PNG")
    
    # Get the byte data from the BytesIO object
    img_bytes = buffered.getvalue()
    
    # Encode the byte data to base64
    img_base64 = base64.b64encode(img_bytes)
    
    # Convert the base64 encoded bytes to a string
    img_base64_str = img_base64.decode('utf-8')
    
    return img_base64_str

def byte2base64(byte_img):
    assert byte_img.startswith(b'\xff\xd8\xff\xe0')
    return base64.b64encode(io.BytesIO(byte_img).read()).decode('utf-8') 


def parse_mixed_string(s):
    if s is None:
        return None

    if isinstance(s, dict):
        return s

    if isinstance(s, float) and math.isnan(s):
        return ""

    s = s.strip("```json")
    s = s.strip("```")
    try:
        # Attempt to parse with ast.literal_eval
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    else:
        traceback.print_exc()
        print(s)
    
    try:
        # Attempt to parse with json.loads
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # If both parsing attempts fail, return the original string
    return None

def path_to_base64(img_path):
    image_path = Path(img_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

def path_to_base64_size(image_path, target_size=(512, 512)):
    with Image.open(image_path) as img:
        img = img.resize(target_size)

        # Save image to a buffer in PNG format
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def pillow_to_base64(pillow_img):
    buffer = BytesIO()
    pillow_img.save(buffer, format="PNG")
    buffer.seek(0)
    # Encode to base64
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def session_img_path_to_base64(images_path, resize=True):
    
     # input image
    src_img = Image.open(images_path[0])
    turn1_img = Image.open(images_path[1])
    if resize:
        target_size = turn1_img.size
        src_img = src_img.resize(target_size)

    src_img_base64 = pillow_to_base64(src_img)
    images_base64 = [src_img_base64] + [path_to_base64(p) for p in images_path[1:]]
    return images_base64