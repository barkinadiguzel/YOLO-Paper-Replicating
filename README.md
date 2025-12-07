# 👀 YOLOv1 – Paper Replication

Replication of **YOLO: You Only Look Once – Unified, Real-Time Object Detection** (Redmon et al., 2016). This project reproduces the YOLOv1 model and its real-time detection pipeline as described in the original paper.

**Paper:** [YOLO: You Only Look Once (arXiv 2016)](https://arxiv.org/abs/1506.02640)

---

## 🖼 Overview – Model & Detection Logic

YOLOv1 treats object detection as a **single regression problem**:  

- The input image is divided into an **S × S grid**.  
- Each grid cell predicts a fixed number of **bounding boxes**, **confidence scores**, and **class probabilities**.  
- **End-to-end training** allows the model to simultaneously learn object **localization** and **classification**.  
- This unified approach enables **real-time detection** with reasonable accuracy.

![Figure Overview](images/fig1.png)
*Figure:* YOLOv1 model architecture overview.

---

## 🧮 Key Idea – Prediction Mechanism

- Each bounding box prediction includes **coordinates (x, y, w, h)** and a **confidence score** representing $$(Pr(object) \cdot IOU_{pred}^{truth}\)$$.  
- Each grid cell also predicts **class probabilities** $$(P(Class_i|Object)\)$$.  
- At inference, final score for a class in a box:

$$Score = Pr(Object) \cdot IOU_{pred}^{truth} \cdot P(Class_i|Object)$$

> This formulation allows YOLOv1 to **simultaneously detect multiple objects** while maintaining a simple and fast computation pipeline.

---

## 🖼 Figures

### Figure 2 – Grid and Bounding Boxes
![Grid Cells](images/fig2.png)

### Figure 3 – Convolutional Layers
![Convolutional Layers](images/fig3.png)

### Figure 6 – Detection Results
![Detection Examples](images/fig6.png)

---

## 🏗️ Project Structure

```bash
YOLOv1-Paper-Replicating/
│
├── src/
│   ├── backbone/
│   │   └── conv_block.py
│   ├── detection_head/
│   │   └── yolo_layer.py
│   ├── utils/
│   │   └── grid_utils.py
│   ├── yolo_model.py
│   ├── mns_decode_and_visualize.py
│   └── config.py
│
├── images/
│   ├── fig1.png
│   ├── fig2.png
│   ├── fig3.png
│   └── fig6.png
│
└── requirements.txt

```
## 🔗 Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
