from src.metrics.base_metrics import BaseMetrics

class TI(BaseMetrics):
    def update(self, prediction, gold):
        pred_triggers = self._extract_triggers(prediction)
        gold_triggers = self._extract_triggers(gold)

        pred_set = set(pred_triggers)
        gold_set = set(gold_triggers)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        self.update_counts(tp, fp, fn)

    def _extract_triggers(self, data):
        triggers = []

        for event in data.get("events", []):
            for trigger in event.get("trigger", []):
                triggers.append(tuple(trigger["span"]))

        return triggers