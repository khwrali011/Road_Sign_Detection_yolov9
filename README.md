## **YOLOv9 Model Training Report**
### **Model Overview**
- **Model Type**: YOLOv9-tiny (YOLOv9t)
- **Framework**: Ultralytics YOLOv8.2.93, PyTorch 2.4.0 with CUDA 12.1
- **Device**: Tesla T4 GPU (15,102 MiB)
- **Parameters**: 2,006,188
- **GFLOPs**: 7.9 GFLOPs (indicating the computational complexity)
- **Layers**: 917
- **Pretrained Weights**: Yes, transfer learning enabled. Transferred 1303/1339 items from pretrained weights.

---

### **Dataset Details**
- **Training Dataset**: 701 images (road signs)
- **Validation Dataset**: 176 images
- **Number of Classes**: 4 (Traffic Light, Stop, Speed Limit, Crosswalk)
- **Instances in Training Set**: 243 road signs across all classes
- **Data Augmentations**:
  - **Blur**: Applied with probability `p=0.01`, using blur limits `(3, 7)`
  - **MedianBlur**: Applied with probability `p=0.01`
  - **ToGray**: Converts images to grayscale with probability `p=0.01`
  - **CLAHE**: Contrast Limited Adaptive Histogram Equalization with clip limit `(1, 4.0)` and tile grid size `(8, 8)`

---

### **Training Configuration**
- **Epochs**: 50
- **Batch Size**: 8
- **Image Size**: 640x640
- **Optimizer**: AdamW, with learning rate `lr=0.00125` and `momentum=0.9`. Optimizer and learning rate were automatically selected.
- **AMP (Automatic Mixed Precision)**: Enabled
- **DataLoader Workers**: 2
- **Frozen Layers**: `model.22.dfl.conv.weight`
- **Mosaic Augmentation**: Close mosaic at epoch 10
- **Hyperparameters**: 
  - **Box Loss**: 7.5
  - **Class Loss**: 0.5
  - **DFL Loss**: 1.5
  - **Label Smoothing**: 0.0
  - **Warmup Epochs**: 3.0

---

### **Results**

#### **Final Training Epoch (Epoch 50/50)**
- **Box Loss**: 0.4576
- **Class Loss**: 0.2752
- **DFL Loss**: 0.8501
- **Instances Processed**: 5 per image
- **Inference Speed**: 6.7ms per image
- **Post-Processing Speed**: 3.4ms per image

#### **Performance on Validation Set**
- **Precision**: 0.961
- **Recall**: 0.929
- **mAP@50**: 0.962 (mean Average Precision at 50% IoU)
- **mAP@50-95**: 0.830 (mean Average Precision over IoU thresholds from 50% to 95%)

#### **Class-wise Detection Performance**
- **Traffic Light**:
  - **Precision**: 0.941
  - **Recall**: 0.846
  - **mAP@50**: 0.934
  - **mAP@50-95**: 0.675
- **Stop Sign**:
  - **Precision**: 0.970
  - **Recall**: 0.947
  - **mAP@50**: 0.993
  - **mAP@50-95**: 0.944
- **Speed Limit**:
  - **Precision**: 0.997
  - **Recall**: 1.000
  - **mAP@50**: 0.995
  - **mAP@50-95**: 0.914
- **Crosswalk**:
  - **Precision**: 0.938
  - **Recall**: 0.921
  - **mAP@50**: 0.928
  - **mAP@50-95**: 0.786

---

### **Model Artifacts**
- **Best Model Weights**: `runs/detect/train2/weights/best.pt` (4.6 MB)
- **Last Model Weights**: `runs/detect/train2/weights/last.pt` (4.6 MB)
- **Label Plot**: Saved to `runs/detect/train2/labels.jpg`
- **Training Logs**: Accessible via TensorBoard (`tensorboard --logdir runs/detect/train2`)

---

### **Observations**
1. **High Precision & Recall**: Across all categories, precision and recall were high, especially for Speed Limit signs, with perfect recall (1.000) and near-perfect precision.
2. **Balanced Performance**: mAP@50 for all classes is above 0.90, showing that the model performs well in detecting road signs with reasonable overlap between the bounding boxes and ground truth.
3. **Traffic Light Class**: The mAP@50-95 for traffic lights is relatively lower compared to other classes (0.675), suggesting this class might benefit from further tuning or more data.
4. **Inference Time**: The model runs fast, making it suitable for real-time applications.

---

### **Conclusions**
The YOLOv9-tiny model has been successfully trained for road sign detection, achieving high precision, recall, and mAP scores across all classes. The small model size and fast inference times make this model ideal for deployment in resource-constrained environments, such as embedded systems or mobile devices.