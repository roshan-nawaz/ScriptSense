import cv2
import numpy as np
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from pdf2image import convert_from_path
from PIL import Image as PILImage
from sklearn.cluster import DBSCAN

# Directory to save line images
SAVE_DIR = 'files/test_lines'

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class DetectorRes:
    img: np.ndarray
    bbox: BBox

def detect_lines(img: np.ndarray, kernel_size: int, sigma: float, theta: float, min_area: int) -> List[DetectorRes]:
    """Detect lines in the image."""
    kernel = _compute_kernel(kernel_size, sigma, theta)
    img_filtered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    img_thres = 255 - cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    res = []
    components = cv2.findContours(img_thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in components:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        crop = img[y:y + h, x:x + w]
        res.append(DetectorRes(crop, BBox(x, y, w, h)))
    return res

def _compute_kernel(kernel_size: int, sigma: float, theta: float) -> np.ndarray:
    """Compute anisotropic filter kernel."""
    assert kernel_size % 2  # must be odd size
    half_size = kernel_size // 2
    xs = ys = np.linspace(-half_size, half_size, kernel_size)
    x, y = np.meshgrid(xs, ys)
    sigma_y = sigma
    sigma_x = sigma_y * theta
    exp_term = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
    x_term = (x ** 2 - sigma_x ** 2) / (2 * np.pi * sigma_x ** 5 * sigma_y)
    y_term = (y ** 2 - sigma_y ** 2) / (2 * np.pi * sigma_y ** 5 * sigma_x)
    kernel = (x_term + y_term) * exp_term
    return kernel / np.sum(kernel)

def _cluster_lines(detections: List[DetectorRes], max_dist: float = 0.7, min_words_per_line: int = 1) -> List[List[DetectorRes]]:
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(i, num_bboxes):
            a, b = detections[i].bbox, detections[j].bbox
            if a.y > b.y + b.h or b.y > a.y + a.h:
                continue
            intersection = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            union = a.h + b.h - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = dist_mat[j, i] = 1 - iou

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)
    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])
    return sorted(clustered.values(), key=lambda line: [det.bbox.y + det.bbox.h / 2 for det in line])

def sort_multiline(detections: List[DetectorRes], max_dist: float = 0.7, min_words_per_line: int = 2) -> List[List[DetectorRes]]:
    lines = _cluster_lines(detections, max_dist, min_words_per_line)
    return [sorted(line, key=lambda det: det.bbox.x + det.bbox.w / 2) for line in lines]

def process_pdf(pdf_path: str):
    images = convert_from_path(pdf_path, poppler_path=r'C:/Program Files/Poppler/Library/bin')
    for page_number, image in enumerate(images):
        img = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        detections = detect_lines(img, kernel_size=3, sigma=11, theta=7, min_area=50)
        lines = sort_multiline(detections)
        for line_idx, line in enumerate(lines):
            x_min = min(det.bbox.x for det in line)
            x_max = max(det.bbox.x + det.bbox.w for det in line)
            y_min = min(det.bbox.y for det in line)
            y_max = max(det.bbox.y + det.bbox.h for det in line)

            line_image = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8) * 255
            for det in line:
                x, y, w, h = det.bbox.x - x_min, det.bbox.y - y_min, det.bbox.w, det.bbox.h
                line_image[y:y + h, x:x + w] = det.img

            output_path = f'{SAVE_DIR}/pdf_page{page_number + 1}_line{line_idx + 1}.png'
            PILImage.fromarray(line_image).save(output_path)

def process_image(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    detections = detect_lines(img, kernel_size=3, sigma=11, theta=7, min_area=50)
    lines = sort_multiline(detections)
    for line_idx, line in enumerate(lines):
        x_min = max(0, min(det.bbox.x for det in line) - 10)
        x_max = max(det.bbox.x + det.bbox.w for det in line) + 10
        y_min = max(0, min(det.bbox.y for det in line) - 10)
        y_max = max(det.bbox.y + det.bbox.h for det in line) + 10

        line_image = np.ones((y_max - y_min, x_max - x_min), dtype=np.uint8) * 255
        for det in line:
            x, y, w, h = det.bbox.x - x_min, det.bbox.y - y_min, det.bbox.w, det.bbox.h
            line_image[y:y + h, x:x + w] = det.img

        output_path = f'{SAVE_DIR}/img_line{line_idx + 1}.png'
        PILImage.fromarray(line_image).save(output_path)

def line_split(input_path):
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    if input_path.lower().endswith('.pdf'):
        process_pdf(input_path)
    else:
        process_image(input_path)
    
    print('Successfully completed line splitting.')

    return SAVE_DIR

if __name__ == '__main__':
    sample_path = ''  # Update with the actual path if needed
    line_split(sample_path)
