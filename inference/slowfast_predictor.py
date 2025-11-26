import time
import torch
from inference.base_predictor import BasePredictor

class SlowFastPredictor(BasePredictor):

    def __init__(self, model, classnames, device):
        self.model = model
        self.classnames = classnames
        self.device = device

    def predict(self, pathways):
        """
        pathways = [slow_tensor, fast_tensor]
        """
        slow, fast = pathways
        slow = slow.to(self.device)
        fast = fast.to(self.device)

        start = time.time()
        with torch.no_grad():
            out = self.model([slow, fast])   # SlowFast expects a list
            probs = torch.softmax(out, dim=1)[0]
            topk = torch.topk(probs, 2)
        end = time.time()

        results = [
            {"class": self.classnames[int(idx)], "score": float(score)}
            for idx, score in zip(topk.indices, topk.values)
        ]

        inference_time = end - start

        return results, inference_time
