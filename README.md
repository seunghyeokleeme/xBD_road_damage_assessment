<div align="center">
<h1 align="center">xBD Road Damage Assessment</h1>

<h3>Deep Learning-Based Detection and Assessment of Road Damage Caused by Disaster with Satellite Imagery</h3>


[Jungeun Cha](https://orcid.org/0009-0000-0579-6405)<sup>1,†</sup>, [Seunghyeok Lee](https://orcid.org/0009-0009-4709-7931)<sup>2, †</sup>, [Hoe-Kyoung Kim](https://orcid.org/0000-0001-5057-2558)<sup>3, *</sup>


<sup>1</sup> Department of Urban Planning and Landscape Architecture, Dong-A University, <sup>2</sup> Department of Electronic Engineering, Dong-A University, <sup>3</sup> Department of Urban Planning and Engineering, Dong-A University, <sup>†</sup> These authors contributed equally to this work

</div>

## 🛰️ Project Overview

This project implements a road damage assessment model using the xBD dataset.  
While the model primarily focuses on road damage assessment, it also performs road detection (semantic segmentation).
The model uses a deep learning-based network (PyTorch) and applies various preprocessing and data augmentation techniques to enhance learning performance.

## 🔑 Key Features

- **End-to-End Pipeline:** Provides a complete workflow from data preprocessing to model training, evaluation, and inference
- **Modular Design:** Each module (data loader, model, training script, etc.) is independently structured for easy extension
- **User-Friendly Configuration:** Hyperparameters and path settings through configuration files (config.yaml, etc.)
- **GPU Support:** Fast training environment through CUDA support
- **Diverse Model Architectures:** Supports both UNet and SiameseDiffUNet models

## 🧩 Requirements

- Python 3.8 or higher
- PyTorch
- CUDA (for GPU usage)
- Other Python libraries: numpy, opencv-python, albumentations, matplotlib, etc.  
  *(See `pyproject.toml` for details)*

## 🍳 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/seunghyeokleeme/xBD_road_damage_assessment.git
   cd xBD_road_damage_assessment
   ```

2. Create and activate virtual environment (optional):
   ```bash
   # Using Python venv
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

   # Using uv venv
   uv venv
   ```


## 💿 Dataset

This project uses the **xBD dataset**. The dataset should be structured as follows:

**Note:** This project focuses on road segmentation, which requires road labels. Since the official xBD dataset does not provide them, we generated the ground-truth masks using the process detailed below.

```
datasets/
├── train/
│   ├── images/
│   │   ├── xxx_pre_disaster.png
│   │   └── xxx_post_disaster.png
│   └── targets/
│       └── xxx_post_disaster_target.png
├── hold/
│   ├── images/
│   └── targets/
└── test/
    ├── images/
    └── targets/
```

### Dataset Download Methods

1.  Download the dataset from the [official xBD page](https://xview2.org) and generate road labels yourself by following the hybrid labeling strategy outlined below.
2.  (Recommended) Download the pre-labeled road dataset from this [Google Drive link](https://drive.google.com/drive/folders/1Kd329puBn5_Nc_3Lg5READct4Whd7erR).

### Ground-Truth Generation Process

As the xBD dataset lacks official road annotations, a **Hybrid Labeling Strategy** was developed to generate high-quality ground-truth masks. This approach consists of three sequential steps:

1.  **Baseline Generation via Transfer Learning**

    * A U-Net model was first pre-trained using the `DeepGlobe Road Extraction` dataset, which provides high-fidelity, manually annotated road masks.
    * This pre-trained model was then applied to the pre-disaster images from the xBD dataset to infer baseline road labels.

2.  **Semi-Automated Labeling Pipeline**

    * Concurrently, we used `YOLOv9` for initial road detection.
    * The detected regions were then refined using the `Segment Anything Model (SAM)` to generate precise, pixel-level masks, enabling large-scale label generation.

3.  **Hybrid Validation and Quality Enhancement**

    * To ensure label quality, the outputs from both pipelines (1 and 2) were compared.
    * Cases with significant discrepancies were flagged for manual expert review, where the label was either corrected or selected based on fidelity (e.g., choosing the DeepGlobe-informed mask when SAM results were noisy).
    * This hybrid pipeline allowed us to construct a reliable, large-scale road mask dataset while minimizing annotation noise.

The final ground-truth masks generated through this process consist of three distinct pixel-level classes: **background (class 0)**, **undamaged road (class 1)**, and **damaged road (class 2)**. This serves as the foundational ground truth for the training and evaluation of all models in this study.


## 🍚 Usage

### 1. Data Preprocessing
```bash
python3 ./data_read.py
```

### 2. Run TensorBoard
```bash
tensorboard --logdir='./log'
```

### 3. Model Type Selection
You can select the model during training and testing using the `--model_type` option:

- `"UNet"`: Basic UNet model
  - Uses `SingleDiffDataset` class to generate difference images
  - Calculates the difference between pre- and post-disaster images for road damage assessment
- `"SiameseDiffUNet"`: UNet model utilizing Siamese structure
  - Uses `FusionChangeDataset` class to process pre- and post-disaster images separately
  - Extracts and fuses features from both time points using Siamese structure

### 4. Model Training
```bash
python3 ./train.py \
  --lr 1e-3 \
  --batch_size 12 \
  --num_epoch 100 \
  --seed 0 \
  --data_dir "./datasets_512" \
  --ckpt_dir "./checkpoint_v1" \
  --log_dir "./log/exp1" \
  --result_dir "./results_v1" \
  --mode "train" \
  --model_type "UNet" \
  --train_continue "off"
```

### 5. Model Testing
```bash
python3 ./train.py \
  --lr 1e-3 \
  --batch_size 12 \
  --num_epoch 100 \
  --seed 0 \
  --data_dir "./datasets_512" \
  --ckpt_dir "./checkpoint_v1" \
  --log_dir "./log/exp1" \
  --result_dir "./results_v1" \
  --mode "test" \
  --model_type "UNet" \
  --train_continue "off"
```

### 6. Model Evaluation
```bash
python3 ./eval.py \
  --result_dir "./results_v1" \
  --out_fp "./localization_metrics.json"
```

## 🧪 Experiment

### Experimental Model Descriptions

1. **Experiment 1 (Road Detection Semantic Segmentation)**
   - Performs inference on pre- and post-disaster images separately and overlays them for damage assessment
   - Refer to [xBD_road_segmentation](https://github.com/seunghyeokleeme/xBD_road_segmentation)
   - model archarchitecture: ![model1](./result/model1.png)
   ![method1](./result/Method_1.JPG)

2. **Experiment 2 (UNet-based)**
   - Generates difference images using the `SingleDiffDataset` class
   - Difference image generation process:
     1. Load pre-disaster image (pre_image) and post-disaster image (post_image)
     2. Apply transforms to both images
     3. Create difference image by subtracting pre-disaster image from post-disaster image (`diff_image = post_image - pre_image`)
   - Uses the generated difference image as input for 3-class segmentation (background/normal road/damaged road)
   - Meaning of difference image:
     - Values close to zero: Unchanged areas (background or normal road)
     - Larger values: Areas with significant changes (damaged road)
   - model archarchitecture: ![model2](./result/model2.png)
   ![method2](./result/Method_2.JPG)

3. **Experiment 3 (SiameseDiffUNet)**
   - Uses `FusionChangeDataset` class to input pre- and post-disaster images separately
   - Extracts features from both time points in the same space using Siamese structure
   - Performs road damage assessment in the decoder after feature fusion (or difference)
   - model archarchitecture: ![model3](./result/model3.png)
   ![method3](./result/Method_3.JPG)

---

## 🎮 Performance Comparison

![Performance1](./result/performance1.png)
![Performance2](./result/performance2.png)

## ⭐️ Result Visualization

![Result 5](./result/result5.png)
![Result 6](./result/result6.png)
![Result 7](./result/result7.png)


## 🧭 Reproducible experiment

| Model      | Metric      |   Seed 0 |   Seed 1 |   Seed 2 |   Seed 3 |   Mean |   Std. Dev. |
|:-----------|:------------|---------:|---------:|---------:|---------:|-------:|------------:|
| Overlay    | F1b         |    0.868 |    0.866 |    0.874 |    0.87  |  0.869 |       0.003 |
| Overlay    | Precision_b |    0.881 |    0.895 |    0.88  |    0.849 |  0.876 |       0.02  |
| Overlay    | Recall_b    |    0.855 |    0.839 |    0.868 |    0.893 |  0.864 |       0.023 |
| Overlay    | IoU_b       |    0.766 |    0.764 |    0.776 |    0.77  |  0.769 |       0.005 |
| Overlay    | F1s         |    0.26  |    0.26  |    0.262 |    0.261 |  0.261 |       0.001 |
| Difference | F1b         |    0.761 |    0.771 |    0.748 |    0.759 |  0.76  |       0.01  |
| Difference | Precision_b |    0.816 |    0.818 |    0.842 |    0.829 |  0.826 |       0.012 |
| Difference | Recall_b    |    0.713 |    0.73  |    0.672 |    0.699 |  0.704 |       0.024 |
| Difference | IoU_b       |    0.615 |    0.628 |    0.597 |    0.611 |  0.613 |       0.013 |
| Difference | F1d         |    0.526 |    0.554 |    0.479 |    0.529 |  0.522 |       0.031 |
| Difference | Precision_d |    0.633 |    0.619 |    0.657 |    0.645 |  0.638 |       0.016 |
| Difference | Recall_d    |    0.451 |    0.502 |    0.377 |    0.449 |  0.445 |       0.051 |
| Difference | IoU_d       |    0.357 |    0.384 |    0.315 |    0.36  |  0.354 |       0.029 |
| Difference | F1s         |    0.597 |    0.62  |    0.56  |    0.598 |  0.594 |       0.025 |
| Siamese    | F1b         |    0.628 |    0.718 |    0.638 |    0.73  |  0.678 |       0.053 |
| Siamese    | Precision_b |    0.877 |    0.884 |    0.915 |    0.868 |  0.886 |       0.02  |
| Siamese    | Recall_b    |    0.489 |    0.605 |    0.489 |    0.63  |  0.553 |       0.075 |
| Siamese    | IoU_b       |    0.457 |    0.56  |    0.468 |    0.575 |  0.515 |       0.061 |
| Siamese    | F1d         |    0.271 |    0.432 |    0.318 |    0.477 |  0.374 |       0.096 |
| Siamese    | Precision_d |    0.814 |    0.776 |    0.776 |    0.714 |  0.77  |       0.041 |
| Siamese    | Recall_d    |    0.163 |    0.299 |    0.2   |    0.358 |  0.255 |       0.09  |
| Siamese    | IoU_d       |    0.157 |    0.275 |    0.189 |    0.313 |  0.234 |       0.073 |
| Siamese    | F1s         |    0.378 |    0.518 |    0.414 |    0.553 |  0.466 |       0.083 |

## 🌱 How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Modify and improve the code.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the remote repository (`git push origin feature/YourFeature`).
6. Create a Pull Request.


## 📜 Reference

If this dataset or code contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@article{cha2025deep,
      title={Deep Learning-Based Detection and Assessment of Road Damage Caused by Disaster with Satellite Imagery}, 
      author={Jungeun Cha and Seunghyeok Lee and Hoe-Kyoung Kim},
      journal={Appl. Sci.},
      year={2025},
      volume={15},
      issue={14},
      pages={7669},
      url={https://www.mdpi.com/2076-3417/15/14/7669}, 
      doi={10.3390/app15147669},
      note={Submission received: 9 June 2025; Revised: 30 June 2025; Accepted: 7 July 2025; Published: 8 July 2025 (This article belongs to the Special Issue Remote Sensing Image Processing and Application, 2nd Edition)},
      abstract={Natural disasters can cause severe damage to critical infrastructure such as road networks, significantly delaying rescue and recovery efforts. Conventional road damage assessments rely heavily on manual inspection, which is labor-intensive, time-consuming, and infeasible in large-scale disaster-affected areas. This study aims to propose a deep learning-based framework to automatically detect and quantitatively assess road damage using high-resolution pre- and post-disaster satellite imagery. To achieve this, the study systematically compares three distinct change detection approaches: single-timeframe overlay, difference-based segmentation, and Siamese feature fusion. Experimental results, validated over multiple runs, show the difference-based model achieved the highest overall F1-score (0.594 ± 0.025), surpassing the overlay and Siamese models by approximately 127.6% and 27.5%, respectively. However, a key finding of this study is that even this best-performing model is constrained by a low detection recall (0.445 ± 0.051) for the 'damaged road' class. This reveals that severe class imbalance is a fundamental hurdle in this domain for which standard training strategies are insufficient. This study establishes a crucial benchmark for the field, highlighting that future research must focus on methods that directly address class imbalance to improve detection recall. Despite its quantified limitations, the proposed framework enables the visualization of damage density maps, supporting emergency response strategies such as prioritizing road restoration and accessibility planning in disaster-stricken areas.},
      keywords={road damage detection and assessment; satellite imagery; deep learning; semantic segmentation; infrastructure resilience}
}
```
