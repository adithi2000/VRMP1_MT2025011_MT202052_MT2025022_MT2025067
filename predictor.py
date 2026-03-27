"""
predictor.py — Student inference file for hidden evaluation.

╔══════════════════════════════════════════════════════════════════╗
║  DO NOT RENAME ANY FUNCTION.                                    ║
║  DO NOT CHANGE FUNCTION SIGNATURES.                             ║
║  DO NOT REMOVE ANY FUNCTION.                                    ║
║  DO NOT RENAME CLS_CLASS_MAPPING or SEG_CLASS_MAPPING.          ║
║  You may add helper functions / imports as needed.              ║
╚══════════════════════════════════════════════════════════════════╝

Tasks
-----
  Task 3.1  — Multi-label image-level classification (5 classes).
  Task 3.2  — Object detection + instance segmentation (5 classes).

You must implement ALL FOUR functions below.

Class Mappings
--------------
  Fill in the two dictionaries below (CLS_CLASS_MAPPING, SEG_CLASS_MAPPING)
  to map your model's output indices to the canonical category names.

  The canonical 5 categories (from the DeepFashion2 subset) are:
      short sleeve top, long sleeve top, trousers, shorts, skirt

  Your indices can be in any order, but the category name strings
  must match exactly (case-insensitive). Background class is optional
  but recommended for detection/segmentation models — the evaluator
  will automatically ignore it.

  Important: Masks must be at the ORIGINAL image resolution.
  If your model internally resizes images, resize the masks back
  to the input image dimensions before returning them.

Model Weights
-------------
  Place your trained weights inside  model_files/  as:
      model_files/cls.pt   (or cls.pth)   — classification model
      model_files/seg.pt   (or seg.pth)   — detection + segmentation model

Evaluation Metrics
------------------
  Classification : Macro F1-score  +  Per-label macro accuracy
  Detection      : mAP @ [0.5 : 0.05 : 0.95]
  Segmentation   : Per-class mIoU (macro-averaged)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import os
import torchvision.models as models
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torchvision.transforms as T


# ═══════════════════════════════════════════════════════════════════
# CLASS MAPPINGS — FILL THESE IN
# ═══════════════════════════════════════════════════════════════════

# Classification: maps your model's output index → canonical class name.
# Must have exactly 5 entries (one per clothing class, NO background).
# Example:
#   CLS_CLASS_MAPPING = {
#       0: "short sleeve top",
#       1: "long sleeve top",
#       2: "trousers",
#       3: "shorts",
#       4: "skirt",
#   }
CLS_CLASS_MAPPING: Dict[int, str] = {
     0: "long sleeve top",
     1: "short sleeve top",
     2: "shorts",
     3: "skirt",
     4: "trousers",
}

# Detection + Segmentation: maps your model's output index → class name.
# Include background if your model outputs it (evaluator will ignore it).
# Example:
#   SEG_CLASS_MAPPING = {
#       0: "background",
#       1: "short sleeve top",
#       2: "long sleeve top",
#       3: "trousers",
#       4: "shorts",
#       5: "skirt",
#   }
#class_names=['background','short sleeve top','trousers','shorts','long sleeve top','skirt']
#load yours accordingly=> if yolo seg trained
SEG_CLASS_MAPPING: Dict[int, str] = {
    0: "background",
    1: "short sleeve top",
    2: "trousers",
    3: "shorts",
    4: "long sleeve top",
    5: "skirt"
}


# ═══════════════════════════════════════════════════════════════════
# Helper utilities (you may modify or add more)
# ═══════════════════════════════════════════════════════════════════

def _find_weights(folder: Path, stem: str) -> Path:
    """Return the first existing weights file matching stem.pt or stem.pth."""
    for ext in (".pt", ".pth"):
        candidate = folder / "model_files" / (stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No weights file found for '{stem}' in {folder / 'model_files'}"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# TASK 3.1 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def load_classification_model(folder: str, device: str) -> Any:
    """
    Load your trained classification model.

    Parameters
    ----------
    folder : str
        Absolute path to your submission folder (the one containing
        this predictor.py, model_files/, class_mapping_cls.json, etc.).
    device : str
        PyTorch device string, e.g. "cuda", "mps", or "cpu".

    Returns
    -------
    model : Any
        Whatever object your predict_classification function needs.
        This is passed directly as the first argument to
        predict_classification().

    Notes
    -----
    - Load weights from  <folder>/model_files/cls.pt  (or .pth).
    - Use CLS_CLASS_MAPPING defined above to map output indices.
    - The returned object can be a dict, a nn.Module, or anything
      your prediction function expects.
    """

    #path
    folder=Path(folder)
    model_path=_find_weights(folder,"cls")
    #rebuilding model architecture
    model=models.resnet50(weights=None)
    model.fc=nn.Linear(model.fc.in_features,5)

    #Load weights
    state_dict=torch.load(model_path,map_location=device)
    model.load_state_dict(state_dict)
    #Move to device
    model=model.to(device)
    #evaluation mode
    model.eval()
    return model
    #raise NotImplementedError("TODO: implement load_classification_model")


def predict_classification(model: Any, images: List[Image.Image]) -> List[Dict]:
    """
    Run multi-label classification on a list of images.

    Parameters
    ----------
    model : Any
        The object returned by load_classification_model().
    images : list of PIL.Image.Image
        A list of RGB PIL images.

    Returns
    -------
    results : list of dict
        One dict per image, with the key "labels":

        [
            {"labels": [int, int, int, int, int]},
            {"labels": [int, int, int, int, int]},
            ...
        ]

        Each "labels" list has exactly 5 elements (one per class,
        in the order defined by your CLS_CLASS_MAPPING dictionary).
        Each element is 0 or 1.

    Example
    -------
    >>> results = predict_classification(model, [img1, img2])
    >>> results[0]
    {"labels": [1, 0, 0, 1, 0]}
    """
    device=next(model.parameters()).device
    transform=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    results=[]
    model.eval()
    with torch.no_grad():
        for img in images:
            image=transform(img)
            image=image.unsqueeze(0).to(device)
            #forward pass
            outputs=model(image)
            #sigmoid
            probs=torch.sigmoid(outputs)[0]
            preds=(probs>0.5).int().cpu().tolist()
            results.append({"labels":preds})
    return results
    #raise NotImplementedError("TODO: implement predict_classification")


# ═══════════════════════════════════════════════════════════════════
# TASK 3.2 — DETECTION + INSTANCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def load_detection_model(folder: str, device: str) -> Any:
    """
    Load your trained detection + segmentation model.

    Parameters
    ----------
    folder : str
        Absolute path to your submission folder.
    device : str
        PyTorch device string, e.g. "cuda", "mps", or "cpu".

    Returns
    -------
    model : Any
        Whatever object your predict_detection_segmentation function
        needs. Passed directly as the first argument.

    Notes
    -----
    - Load weights from  <folder>/model_files/seg.pt  (or .pth).
    - Use SEG_CLASS_MAPPING defined above to map output indices.
    """
    get_weights_path = _find_weights(Path(folder), "seg")
    print(f"Loading detection + segmentation model from {get_weights_path}...")
    num_classes = len(SEG_CLASS_MAPPING)

    model=torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT",min_size=512,
        max_size=512)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    model.rpn.post_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_test = 1000
    model.roi_heads.detections_per_img = 300
    model.roi_heads.score_thresh = 0.3

    state_dict = torch.load(get_weights_path, map_location=device)

    
   
    model.load_state_dict(state_dict)
    
    print(f"Model loaded successfully {('Success' if model is not None else 'Failed')}.")

    model.to(device)
    model.eval()

    return model



def predict_detection_segmentation(
    model: Any,
    images: List[Image.Image],
) -> List[Dict]:
    """
    Run detection + instance segmentation on a list of images.

    Parameters
    ----------
    model : Any
        The object returned by load_detection_model().
    images : list of PIL.Image.Image
        A list of RGB PIL images.

    Returns
    -------
    results : list of dict
        One dict per image with keys "boxes", "scores", "labels", "masks":

        [
            {
                "boxes":  [[x1, y1, x2, y2], ...],   # list of float coords
                "scores": [float, ...],               # confidence in [0, 1]
                "labels": [int, ...],                 # class indices (see mapping)
                "masks":  [np.ndarray, ...]           # binary masks, H×W, uint8
            },
            ...
        ]

    Output contract
    ---------------
    - boxes / scores / labels / masks must all have the same length
      (= number of detected instances in that image).
    - Each box is [x1, y1, x2, y2] with x1 < x2, y1 < y2.
    - Coordinates must be within image bounds (0 ≤ x ≤ width, 0 ≤ y ≤ height).
    - Each score is a float in [0, 1].
    - Each label is an int index matching your SEG_CLASS_MAPPING.
    - Each mask is a 2-D numpy array of shape (image_height, image_width)
      with dtype uint8, containing only 0 and 1.
    - If no objects are detected, return empty lists for all keys.

    Example
    -------
    >>> results = predict_detection_segmentation(model, [img])
    >>> results[0]["boxes"]
    [[100.0, 40.0, 300.0, 420.0], [50.0, 200.0, 250.0, 600.0]]
    >>> results[0]["masks"][0].shape
    (height, width)
    """
    device = next(model.parameters()).device
    results = []
    transform = T.ToTensor()

    tensors = [transform(img).to(device) for img in images]

    with torch.no_grad():
        outputs = model(tensors)

    for img , output in zip(images, outputs):
        boxes = output["boxes"].cpu().numpy().tolist()
        scores = output["scores"].cpu().numpy().tolist()
        labels = output["labels"].cpu().numpy().tolist()
        masks = (output["masks"] > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

        h, w = img.size[1], img.size[0]

        final_boxes = []
        final_scores = []
        final_labels = []
        final_masks = []

        for i in range(len(scores)):

            score = float(scores[i])


            if score < 0.3:
                continue

            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            final_boxes.append([float(x1), float(y1), float(x2), float(y2)])
            final_scores.append(score)
            final_labels.append(int(labels[i]))
            
            mask = masks[i, 0]
            mask = (mask > 0.5).astype(np.uint8)

            final_masks.append(mask)

        results.append({
            "boxes": final_boxes,
            "scores": final_scores, 
            "labels": final_labels,
            "masks": final_masks
        })

    return results

# if __name__ == "__main__":
#     # Example usage:
#     # folder = "D:\New folder\VRMP1_MT2025011_MT2025052_MT2025022_MT2025067\model_files\seg.pth"  # Change to your actual folder path/
#     folder = os.path.dirname(os.path.abspath("./model_files"))
#     print(f"Using folder: {folder}")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Load models
#     # cls_model = load_classification_model(folder, device)
#     seg_model = load_detection_model(folder, device)

#     # Example inference (replace with actual images)
#     dummy_image = Image.new("RGB", (512, 512), color="white")
#     # cls_results = predict_classification(cls_model, [dummy_image])
#     seg_results = predict_detection_segmentation(seg_model, [dummy_image])

#     # print("Classification results:", cls_results)
#     print("Detection + Segmentation results:", seg_results)