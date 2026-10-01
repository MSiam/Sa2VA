import torch
import numpy as np
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
from third_parts.revos.utils.metrics import db_eval_iou, db_eval_boundary

@METRICS.register_module()
class VidRefSegMetric(BaseMetric):
    def __init__(self, prefix=None, collect_device='cpu'):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            gt_masks, pred_masks = sample['gt_masks'], sample['pred_masks']

            pred_masks = torch.sigmoid(pred_masks)
            pred_masks[pred_masks>= 0.5] = 1
            pred_masks[pred_masks< 0.5] = 0

            pred_masks = np.array(torch.tensor(pred_masks.detach(), dtype=torch.uint8).cpu())
            gt_masks = np.array(gt_masks.cpu())

            j = db_eval_iou(gt_masks, pred_masks).mean()
            f = db_eval_boundary(gt_masks, pred_masks).mean()

            self.results.append({
                'video_id_exp': sample['id'],
                'j': j,
                'f': f
            })

    def compute_metrics(self, results):
        j = [x['j'] for x in results]
        f = [x['f'] for x in results]

        results = {
            'J': round(100 * float(np.mean(j)), 2),
            'F': round(100 * float(np.mean(f)), 2),
            'J&F': round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
        }
        return results
