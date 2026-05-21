import numpy as np
import cv2
from PIL import Image
import io
from typing import Tuple, Optional

TARGET_SIZE = (128, 128) 

HAAR_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_PROFILE  = cv2.data.haarcascades + "haarcascade_profileface.xml"

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil_img)
    gray   = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    return gray

def load_image_from_pil(pil_image: Image.Image) -> np.ndarray:
    rgb  = pil_image.convert("RGB")
    np_img = np.array(rgb)
    gray   = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    return gray

def detect_face(gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    face_cascade = cv2.CascadeClassifier(HAAR_FRONTAL)
    
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])
    
    profile_cascade = cv2.CascadeClassifier(HAAR_PROFILE)
    faces = profile_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])
    
    return None  


def crop_face(gray_image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    
    H, W = gray_image.shape
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    
    return gray_image[y1:y2, x1:x2]

def resize_and_normalize(gray_image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    resized = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    normalized = equalized.astype(np.float64) / 255.0
    
    return normalized


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def preprocess_face(
    gray_image: np.ndarray,
    detect: bool = True,
    target_size: Tuple[int, int] = TARGET_SIZE,
    blur: bool = True,
) -> Tuple[np.ndarray, dict]:
    info = {"face_detected": False, "bbox": None}
    
    if detect:
        bbox = detect_face(gray_image)
        if bbox is not None:
            info["face_detected"] = True
            info["bbox"] = bbox
            face_crop = crop_face(gray_image, bbox)
        else:
            face_crop = gray_image
    else:
        face_crop = gray_image
    
    if blur:
        face_crop = apply_gaussian_blur(face_crop)
    
    processed = resize_and_normalize(face_crop, target_size)
    
    return processed, info


def draw_face_box(original_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    if len(original_image.shape) == 2:
        display = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        display = original_image.copy()
    
    x, y, w, h = bbox
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 100), 2)
    
    return display
