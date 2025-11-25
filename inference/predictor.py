# inference/predictor.py
import time
import torch

class Predictor:
    def __init__(self, model, classnames, device):
        self.model = model
        self.classnames = classnames
        self.device = device

    def predict(self, buffer):
        tensor = buffer.to(self.device)
        start = time.time()
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            # probs = torch.sigmoid(out)[0]
            top5 = torch.topk(probs, 1)
        end = time.time()

        results = [
            {"class": self.classnames[int(idx)], "score": float(score)}
            for idx, score in zip(top5.indices, top5.values)
        ]
        inference_time = end - start
        return results, inference_time
