import base64
from io import BytesIO
from PIL import Image

def pil_image_to_base64(img: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG string.
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
