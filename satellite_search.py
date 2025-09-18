# Satellite Visual Search and Detection System with Visual Outputs (Multi-Object Version)
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from skimage.util import view_as_windows
from sklearn.metrics.pairwise import cosine_similarity
import logging
import datetime
from datetime import date
import cv2
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import Image, display
import matplotlib.patches as patches
from datetime import date
import csv

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Feature Extractor with custom normalization
def get_feature_extractor():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use ResNet50 with custom modifications
    model = models.resnet50(weights='DEFAULT')
    # Remove the final classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    
    # Add custom pooling and normalization
    class CustomFeatureExtractor(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.norm = nn.functional.normalize
            
        def forward(self, x):
            features = self.base_model(x)
            pooled = self.global_pool(features)
            flattened = self.flatten(pooled)
            normalized = self.norm(flattened, p=2, dim=1)
            return normalized
    
    model = CustomFeatureExtractor(model)
    model.eval().to(device)
    logging.info(f"Loaded enhanced ResNet50 with custom normalization on {device}")
    return model

# Enhanced Preprocessing specifically for satellite imagery
def preprocess_image(img: np.ndarray, target_size=(224, 224)) -> torch.Tensor:
    """Convert numpy image to normalized tensor with satellite-specific preprocessing."""
    # Convert to LAB color space for better contrast handling
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    # Resize image
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    
    # Use ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor

# Extract Embeddings from Chips with advanced augmentation
def extract_embeddings(chip_paths: List[str], model) -> np.ndarray:
    embeddings = []
    device = next(model.parameters()).device
    
    for chip_path in chip_paths:
        try:
            img_cv = cv2.imread(chip_path, cv2.IMREAD_COLOR)
            if img_cv is None:
                raise ValueError(f"Image not loaded: {chip_path}")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Apply multiple augmentations to improve robustness
            aug_embeddings = []
            
            # Original image
            tensor = preprocess_image(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).cpu().numpy()[0]
            aug_embeddings.append(emb)
            
            # Horizontal flip
            img_flip = cv2.flip(img_rgb, 1)
            tensor = preprocess_image(img_flip).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).cpu().numpy()[0]
            aug_embeddings.append(emb)
            
            # Slight rotation (5 degrees)
            h, w = img_rgb.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 5, 1.0)
            img_rot = cv2.warpAffine(img_rgb, M, (w, h))
            tensor = preprocess_image(img_rot).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).cpu().numpy()[0]
            aug_embeddings.append(emb)
            
            # Brightness adjustment
            img_bright = np.clip(img_rgb * 1.2, 0, 255).astype(np.uint8)
            tensor = preprocess_image(img_bright).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(tensor).cpu().numpy()[0]
            aug_embeddings.append(emb)
            
            # Average all augmented embeddings
            emb = np.mean(aug_embeddings, axis=0)
            embeddings.append(emb)
            logging.info(f"Extracted emb for {chip_path}: shape {emb.shape}")
        except Exception as e:
            logging.error(f"Failed to process {chip_path}: {e}")
    
    if not embeddings:
        raise ValueError("No valid chips loaded.")
    
    # Average and normalize the embeddings
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    logging.info(f"Avg emb shape: {avg_emb.shape}, norm: {np.linalg.norm(avg_emb)}")
    return avg_emb

# NMS with improved threshold
def nms(boxes: np.ndarray, iou_thresh: float = 0.3) -> np.ndarray:
    if len(boxes) == 0: return boxes
    indices = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        if len(indices) == 1: break
        ious = compute_iou(boxes[i], boxes[indices[1:]])
        indices = indices[1:][ious < iou_thresh]
    return boxes[keep]

# Compute IoU
def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-6)

# Create a directory for visual outputs
def setup_visualization_dir(output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_path: str, detections: List[List], output_path: str, object_name: str = "Unknown"):
    """
    Draw bounding boxes on an image and save it.
    
    Args:
        image_path: Path to the original image
        detections: List of detections in format [x_min, y_min, x_max, y_max, score]
        output_path: Path to save the image with bounding boxes
        object_name: Name of the detected object
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not load image: {image_path}")
        return
    
    # Convert to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img_rgb)
    
    # Draw each bounding box
    for det in detections:
        x_min, y_min, x_max, y_max, score = det[:5]
        
        # Create a rectangle patch
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                        linewidth=2, edgecolor='lime', facecolor='none')
        
        # Add the patch to the axis
        ax.add_patch(rect)
        
        # Add label with confidence score
        ax.text(x_min, y_min - 10, f"{object_name}: {score:.2f}", 
                color='white', backgroundcolor='lime', fontsize=10)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"Saved visualization to {output_path}")

# Function to display the image in the notebook
def display_image(image_path: str):
    """Display an image in the notebook."""
    display(Image(filename=image_path))

# Enhanced Detect in Single Image with visualization
def detect_in_image(target_path: str, avg_emb: np.ndarray, model, object_name: str,
                    threshold: float = 0.4, stride_factor: float = 0.25,
                    scales: List[float] = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4], 
                    base_win_size: int = 224, viz_dir: str = "visualizations") -> List[List]:

    try:
        img_cv = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError("OpenCV failed to load target")
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        detections = []
        device = next(model.parameters()).device
        
        # Log image dimensions for debugging
        logging.info(f"Processing {os.path.basename(target_path)} for {object_name}: size {h}x{w}")
        
        # Track max similarity for debugging
        max_sim = 0.0
        
        # Apply global enhancement to the entire image
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels
        img_lab = cv2.merge([l, a, b])
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Process each scale
        for scale in scales:
            win_size = max(32, min(int(base_win_size * scale), min(h, w)))
            stride = int(win_size * stride_factor)
            
            # Skip if window is too small or stride is too small
            if win_size < 32 or stride < 8:
                continue
                
            logging.info(f"Scale {scale}: window size {win_size}, stride {stride}")
            
            # Extract windows
            windows = view_as_windows(img_enhanced, (win_size, win_size, 3), step=(stride, stride, 3))
            num_win_h, num_win_w = windows.shape[:2]
            patches = windows.reshape(-1, win_size, win_size, 3)
            
            batch_size = 32  # Increased batch size for efficiency
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                tensors = torch.stack([preprocess_image(p) for p in batch]).to(device)
                
                with torch.no_grad():
                    embs = model(tensors).cpu().numpy()
                    # Ensure embeddings are 2D
                    if len(embs.shape) > 2:
                        embs = embs.reshape(embs.shape[0], -1)
                    
                    # Normalize embeddings
                    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-6)
                
                # Compute cosine similarity
                sims = cosine_similarity(embs, avg_emb.reshape(1, -1)).flatten()
                
                # Track max similarity
                current_max = np.max(sims)
                if current_max > max_sim:
                    max_sim = current_max
                
                # Find detections above threshold
                for j, sim in enumerate(sims):
                    if sim > threshold:
                        row = (i + j) // num_win_w
                        col = (i + j) % num_win_w
                        x_min, y_min = col * stride, row * stride
                        x_max, y_max = x_min + win_size, y_min + win_size
                        detections.append([x_min, y_min, x_max, y_max, sim])
                        logging.info(f"Match at ({x_min},{y_min}) score {sim:.3f}")
        
        # Log max similarity for debugging
        logging.info(f"Max similarity in {os.path.basename(target_path)} for {object_name}: {max_sim:.4f}")
        
        if detections:
            detections = nms(np.array(detections))
            formatted_detections = [[int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[4])] for d in detections]
            
            # Create visualization
            viz_path = os.path.join(viz_dir, f"detection_{object_name}_{os.path.basename(target_path)}")
            draw_bounding_boxes(target_path, detections, viz_path, object_name)
            
            # Display the image in the notebook
            print(f"\nDetections in {os.path.basename(target_path)} for {object_name}:")
            display_image(viz_path)
            
            return formatted_detections
        return []
    except Exception as e:
        logging.error(f"Error processing {target_path} for {object_name}: {e}")
        return []

# Generate CSV Output
def generate_csv_output(all_detections: Dict[str, List[List]], output_dir: str) -> str:
    today = date.today().strftime("%d-%b-%Y")
    output_path = os.path.join(output_dir, f"detections_{today}.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['x_min', 'y_min', 'x_max', 'y_max', 'searched_object_name', 'target_imagery_file_name', 'similarity_score'])
        
        # Write all detections
        for object_name, detections in all_detections.items():
            for det in detections:
                # Format: [x_min, y_min, x_max, y_max, score, image_name]
                row = [det[0], det[1], det[2], det[3], object_name, det[5], det[4]]
                writer.writerow(row)
    
    logging.info(f"CSV output saved to {output_path}")
    return output_path

# Main Function with visualization for multiple objects
def run_multi_object_visual_search(chip_base_dir: str, target_folder: str, output_dir: str, 
                                  object_names: List[str], group_name: str, 
                                  threshold: float = 0.35):
    # Setup visualization directory
    viz_dir = setup_visualization_dir()
    
    # Load model
    model = get_feature_extractor()
    
    # Get target images
    target_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(target_folder) if any(f.lower().endswith(ext) for ext in target_extensions)]
    logging.info(f"Loaded {len(image_files)} target images from {target_folder}")
    
    # Process each object
    all_detections = {}
    
    # Display chip images for reference
    print("\n=== REFERENCE CHIP IMAGES ===")
    
    for object_name in object_names:
        # Get chip paths for this object
        chip_folder = os.path.join(chip_base_dir, object_name)
        if not os.path.exists(chip_folder):
            logging.warning(f"Chip folder not found for {object_name}: {chip_folder}")
            continue
            
        chip_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        chip_paths = [os.path.join(chip_folder, f) for f in os.listdir(chip_folder) 
                     if any(f.lower().endswith(ext) for ext in chip_extensions)]
        
        if len(chip_paths) > 5 or len(chip_paths) == 0:
            logging.warning(f"Chip folder for {object_name} must contain 1-5 images. Found {len(chip_paths)}.")
            continue
            
        logging.info(f"Processing {object_name} with {len(chip_paths)} chips")
        
        # Display chip images for this object
        print(f"\nChips for {object_name}:")
        for chip_path in chip_paths:
            display_image(chip_path)
        
        # Extract average embedding for this object
        avg_emb = extract_embeddings(chip_paths, model)
        
        # Run detection on all targets for this object
        object_detections = []
        
        print(f"\n=== DETECTION RESULTS FOR {object_name.upper()} ===")
        
        for file in image_files:
            path = os.path.join(target_folder, file)
            dets = detect_in_image(path, avg_emb, model, object_name, threshold=threshold, viz_dir=viz_dir)
            
            # Add image name to each detection
            for det in dets:
                det.append(file)  # Add image name at the end
            
            object_detections.extend(dets)
            logging.info(f"Detections for {object_name} in {file}: {len(dets)}")
        
        all_detections[object_name] = object_detections
    
    # Generate CSV output
    output_path = generate_csv_output(all_detections, output_dir)
    total_detections = sum(len(d) for d in all_detections.values())
    return f"Done! Output: {output_path} | Total Detections: {total_detections} | Visualizations saved in {viz_dir}"

# Example Usage
if __name__ == "__main__":
    # Customize these paths
    chip_base_dir = "chips"  # Base directory with subfolders for each object
    target_folder = "img_dataset/mock"  # Folder with images to search
    output_dir = "outputs_final"  # Output folder for CSV file
    object_names = ["metroshed", "Stp", "sheds", "brick kiln", "pond", "playground", "solar panel"]  # List of objects to search for
    group_name = "test1"
    threshold = 0.80  # Similarity threshold
    
    # Run with visualization
    result = run_multi_object_visual_search(chip_base_dir, target_folder, output_dir, object_names, group_name, threshold)
    print(result)
