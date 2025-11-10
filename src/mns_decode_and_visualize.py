import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def nms(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []
    
    # detections: [(x1, y1, x2, y2, score, class_idx), ...]
    detections = sorted(detections, key=lambda x: x[4], reverse=True)  
    keep = []

    while detections:
        best = detections.pop(0)
        keep.append(best)
        filtered = []
        for det in detections:
            if best[5] != det[5]:  
                filtered.append(det)
                continue
            # IoU 
            x1 = max(best[0], det[0])
            y1 = max(best[1], det[1])
            x2 = min(best[2], det[2])
            y2 = min(best[3], det[3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            box1_area = (best[2]-best[0]) * (best[3]-best[1])
            box2_area = (det[2]-det[0]) * (det[3]-det[1])
            iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
            if iou <= iou_threshold:
                filtered.append(det)
        detections = filtered
    return keep

def decode_predictions(pred_tensor, S=7, B=2, C=20, conf_threshold=0.2, iou_threshold=0.5):
    batch_size = pred_tensor.size(0)
    all_detections = []

    for i in range(batch_size):
        detections = []
        for row in range(S):
            for col in range(S):
                cell = pred_tensor[i, row, col]
                boxes = cell[:B*5].view(B,5)
                class_probs = cell[B*5:]

                for b in range(B):
                    x, y, w, h, conf = boxes[b]
                    scores = conf * class_probs
                    class_idx = torch.argmax(scores)
                    score = scores[class_idx]

                    if score > conf_threshold:
                        x_center = (col + x) / S
                        y_center = (row + y) / S
                        x1 = x_center - w/2
                        y1 = y_center - h/2
                        x2 = x_center + w/2
                        y2 = y_center + h/2
                        detections.append((x1.item(), y1.item(), x2.item(), y2.item(), score.item(), class_idx.item()))
        
        # apply NMS 
        detections = nms(detections, iou_threshold)
        all_detections.append(detections)
    return all_detections

def plot_detections(image, all_detections, batch_idx=0):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for det in all_detections[batch_idx]:
        x1, y1, x2, y2, score, class_idx = det
        x1 *= image.shape[1]
        x2 *= image.shape[1]
        y1 *= image.shape[0]
        y2 *= image.shape[0]
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{class_idx}: {score:.2f}', color='yellow', fontsize=8)
    plt.show()
