class BaseMetrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update_counts(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def precision(self):
        if self.tp + self.fp == 0:
            return 0.0

        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0.0

        return self.tp / (self.tp + self.fn)

    def f1(self):
        p = self.precision()
        r = self.recall()

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def compute(self):
        return {
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0