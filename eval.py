#!/usr/bin/env python
import os
import argparse
from PIL import Image
import numpy as np
import json

def compute_localization_metrics(result_dir):
    """
    Computes various metrics for road detection and damage assessment
    using semantic segmentation results from images saved in the png folder.
    
    1) Road segmentation evaluation (F1b):
       - Reads the saved mask and prediction RGB images.
       - Maps RGB colors to class indices: [0,0,0] -> 0 (background),
         [0,255,0] -> 1 (normal road), [255,0,0] -> 2 (damaged road).
       - Then, the binary masks are created by treating classes 1 and 2 as road (1),
         and background as 0. Pixel-wise TP, FP, FN, TN are computed.
    
    2) Road damage evaluation (F1d):
       - Within the road area, damaged road (class 2) is considered positive,
         while normal road (class 1) is considered negative.
    
    3) Damage F1 per class:
       - Computes the F1 score for each class (0, 1, 2) using one-vs-all evaluation.
    
    The overall score (F1s) is defined as: F1s = 0.3 * F1b + 0.7 * F1d.
    
    Returns:
        metrics (dict): {
            'precision_local': ...,
            'recall_local': ...,
            'f1_local': ...,
            'accuracy_local': ...,
            'iou_local': ...,
            'precision_damage': ...,
            'recall_damage': ...,
            'f1_damage': ...,
            'accuracy_damage': ...,
            'iou_damage': ...,
            'damage_f1_per_class': {'class_0': ..., 'class_1': ..., 'class_2': ...},
            'F1s': overall F1 score (0.3 * F1b + 0.7 * F1d),
            'TP_local': ...,
            'FP_local': ...,
            'FN_local': ...,
            'TN_local': ...,
            'TP_damage': ...,
            'FP_damage': ...,
            'FN_damage': ...,
            'TN_damage': ...
        }
    """
    png_dir = os.path.join(result_dir, 'png')
    
    # Create and sort the list of files
    lst_images = sorted([f for f in os.listdir(png_dir) if f.startswith('test_input')])
    lst_labels = sorted([f for f in os.listdir(png_dir) if f.startswith('test_mask')])
    lst_preds = sorted([f for f in os.listdir(png_dir) if f.startswith('test_pred')])
    
    print(f"Number of images: {len(lst_images)}")
    print(f"Number of labels: {len(lst_labels)}")
    print(f"Number of preds: {len(lst_preds)}")
    
    # Initialize accumulators for road segmentation metrics (F1b)
    total_TP_local, total_FP_local, total_FN_local, total_TN_local = 0, 0, 0, 0
    # Initialize accumulators for road damage evaluation metrics (F1d)
    total_TP_damage, total_FP_damage, total_FN_damage, total_TN_damage = 0, 0, 0, 0

    # For Damage F1 per class: a dictionary to store TP, FP, FN for each class
    class_metrics = {0: {"TP": 0, "FP": 0, "FN": 0},
                     1: {"TP": 0, "FP": 0, "FN": 0},
                     2: {"TP": 0, "FP": 0, "FN": 0}}
    
    for label_file, pred_file in zip(lst_labels, lst_preds):
        # Open RGB mask files
        label_rgb = np.array(Image.open(os.path.join(png_dir, label_file)).convert('RGB'))
        pred_rgb = np.array(Image.open(os.path.join(png_dir, pred_file)).convert('RGB'))
        
        # Map RGB values to class indices:
        # [0,0,0] -> 0 (background)
        # [0,255,0] -> 1 (normal road)
        # [255,0,0] -> 2 (damaged road)
        label_class = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
        pred_class = np.zeros(pred_rgb.shape[:2], dtype=np.uint8)
        
        # For normal road: if green channel is 255 and others are 0, assign 1
        label_class[np.where((label_rgb[:,:,0] == 0) & (label_rgb[:,:,1] == 255) & (label_rgb[:,:,2] == 0))] = 1
        pred_class[np.where((pred_rgb[:,:,0] == 0) & (pred_rgb[:,:,1] == 255) & (pred_rgb[:,:,2] == 0))] = 1
        
        # For damaged road: if red channel is 255 and others are 0, assign 2
        label_class[np.where((label_rgb[:,:,0] == 255) & (label_rgb[:,:,1] == 0) & (label_rgb[:,:,2] == 0))] = 2
        pred_class[np.where((pred_rgb[:,:,0] == 255) & (pred_rgb[:,:,1] == 0) & (pred_rgb[:,:,2] == 0))] = 2
        
        # 1) Road segmentation evaluation (F1b): Treat road as any pixel with class > 0
        label_bin = (label_class > 0).astype(np.uint8)
        pred_bin = (pred_class > 0).astype(np.uint8)
        
        TP_local = ((pred_bin == 1) & (label_bin == 1)).sum()
        FN_local = ((pred_bin == 0) & (label_bin == 1)).sum()
        FP_local = ((pred_bin == 1) & (label_bin == 0)).sum()
        TN_local = ((pred_bin == 0) & (label_bin == 0)).sum()
        
        total_TP_local += TP_local
        total_FP_local += FP_local
        total_FN_local += FN_local
        total_TN_local += TN_local
        
        # 2) Road damage evaluation (F1d): Within road pixels, consider damaged road (class 2) as positive
        label_damage = (label_class == 2).astype(np.uint8)
        pred_damage = (pred_class == 2).astype(np.uint8)
        
        TP_damage = ((pred_damage == 1) & (label_damage == 1)).sum()
        FN_damage = ((pred_damage == 0) & (label_damage == 1)).sum()
        FP_damage = ((pred_damage == 1) & (label_damage == 0)).sum()
        TN_damage = ((pred_damage == 0) & (label_damage == 0)).sum()
        
        total_TP_damage += TP_damage
        total_FP_damage += FP_damage
        total_FN_damage += FN_damage
        total_TN_damage += TN_damage
        
        # 3) Damage F1 per class: Compute one-vs-all F1 score for each class (0,1,2)
        for cls in [0, 1, 2]:
            label_cls = (label_class == cls).astype(np.uint8)
            pred_cls = (pred_class == cls).astype(np.uint8)
            
            TP_cls = ((pred_cls == 1) & (label_cls == 1)).sum()
            FP_cls = ((pred_cls == 1) & (label_cls == 0)).sum()
            FN_cls = ((pred_cls == 0) & (label_cls == 1)).sum()
            
            class_metrics[cls]["TP"] += TP_cls
            class_metrics[cls]["FP"] += FP_cls
            class_metrics[cls]["FN"] += FN_cls

    # Compute metrics for road segmentation (F1b)
    precision_local = total_TP_local / (total_TP_local + total_FP_local) if (total_TP_local + total_FP_local) > 0 else 0
    recall_local = total_TP_local / (total_TP_local + total_FN_local) if (total_TP_local + total_FN_local) > 0 else 0
    f1_local = (2 * precision_local * recall_local) / (precision_local + recall_local) if (precision_local + recall_local) > 0 else 0
    accuracy_local = (total_TP_local + total_TN_local) / (total_TP_local + total_FP_local + total_FN_local + total_TN_local) if (total_TP_local + total_FP_local + total_FN_local + total_TN_local) > 0 else 0
    iou_local = total_TP_local / (total_TP_local + total_FP_local + total_FN_local) if (total_TP_local + total_FP_local + total_FN_local) > 0 else 0
    
    # Compute metrics for road damage evaluation (F1d) using class 2
    precision_damage = total_TP_damage / (total_TP_damage + total_FP_damage) if (total_TP_damage + total_FP_damage) > 0 else 0
    recall_damage = total_TP_damage / (total_TP_damage + total_FN_damage) if (total_TP_damage + total_FN_damage) > 0 else 0
    f1_damage = (2 * precision_damage * recall_damage) / (precision_damage + recall_damage) if (precision_damage + recall_damage) > 0 else 0
    accuracy_damage = (total_TP_damage + total_TN_damage) / (total_TP_damage + total_FP_damage + total_FN_damage + total_TN_damage) if (total_TP_damage + total_FP_damage + total_FN_damage + total_TN_damage) > 0 else 0
    iou_damage = total_TP_damage / (total_TP_damage + total_FP_damage + total_FN_damage) if (total_TP_damage + total_FP_damage + total_FN_damage) > 0 else 0

    # Compute Damage F1 per class (one-vs-all F1 score for each class)
    damage_f1_per_class = {}
    for cls in [0, 1, 2]:
        TP_cls = class_metrics[cls]["TP"]
        FP_cls = class_metrics[cls]["FP"]
        FN_cls = class_metrics[cls]["FN"]
        f1_cls = (2 * TP_cls) / (2 * TP_cls + FP_cls + FN_cls) if (2 * TP_cls + FP_cls + FN_cls) > 0 else 0
        damage_f1_per_class[f"class_{cls}"] = f1_cls

    # Overall score: F1s = 0.3 * f1_local + 0.7 * f1_damage
    f1s = 0.3 * f1_local + 0.7 * f1_damage

    metrics = {
        "precision_local": precision_local,
        "recall_local": recall_local,
        "f1_local": f1_local,
        "accuracy_local": accuracy_local,
        "iou_local": iou_local,
        "precision_damage": precision_damage,
        "recall_damage": recall_damage,
        "f1_damage": f1_damage,
        "accuracy_damage": accuracy_damage,
        "iou_damage": iou_damage,
        "damage_f1_per_class": damage_f1_per_class,
        "F1s": f1s,
        "TP_local": int(total_TP_local),
        "FP_local": int(total_FP_local),
        "FN_local": int(total_FN_local),
        "TN_local": int(total_TN_local),
        "TP_damage": int(total_TP_damage),
        "FP_damage": int(total_FP_damage),
        "FN_damage": int(total_FN_damage),
        "TN_damage": int(total_TN_damage)
    }
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute localization (road detection) and damage assessment metrics for semantic segmentation results'
    )
    parser.add_argument('--result_dir', type=str, default='./results_v1',
                        help='Path to results directory (default: ./results_v1)')
    parser.add_argument('--out_fp', type=str, default='localization_metrics.json',
                        help='Output JSON file path (default: localization_metrics.json)')
    args = parser.parse_args()
    
    metrics = compute_localization_metrics(args.result_dir)
    
    print("Localization Metrics (F1b - road segmentation):")
    print(f"Precision: {metrics['precision_local']:.4f}")
    print(f"Recall   : {metrics['recall_local']:.4f}")
    print(f"F1 Score : {metrics['f1_local']:.4f}")
    print(f"Accuracy : {metrics['accuracy_local']:.4f}")
    print(f"IoU      : {metrics['iou_local']:.4f}")
    
    print("\nDamage Assessment Metrics (F1d - binary damage evaluation):")
    print(f"Precision: {metrics['precision_damage']:.4f}")
    print(f"Recall   : {metrics['recall_damage']:.4f}")
    print(f"F1 Score : {metrics['f1_damage']:.4f}")
    print(f"Accuracy : {metrics['accuracy_damage']:.4f}")
    print(f"IoU      : {metrics['iou_damage']:.4f}")
    
    print("\nDamage F1 per class:")
    for cls, f1_val in metrics['damage_f1_per_class'].items():
        print(f"{cls}: {f1_val:.4f}")
    
    print(f"\nOverall F1 Score (F1s = 0.3 * F1b + 0.7 * F1d): {metrics['F1s']:.4f}")
    print(f"TP_local: {metrics['TP_local']}, FP_local: {metrics['FP_local']}, FN_local: {metrics['FN_local']}, TN_local: {metrics['TN_local']}")
    print(f"TP_damage: {metrics['TP_damage']}, FP_damage: {metrics['FP_damage']}, FN_damage: {metrics['FN_damage']}, TN_damage: {metrics['TN_damage']}")
    
    with open(args.out_fp, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {args.out_fp}")
