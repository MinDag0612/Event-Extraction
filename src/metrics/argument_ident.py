
from src.metrics.base_metrics import BaseMetrics


class AI(BaseMetrics):
    """
    Argument Identification

    An argument is correct if:
    - argument span matches
    - event type matches

    Argument role is ignored.
    """

    def update(self, prediction, gold):
        pred_arguments = self._extract_arguments(prediction)
        gold_arguments = self._extract_arguments(gold)

        pred_set = set(pred_arguments)
        gold_set = set(gold_arguments)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        self.update_counts(tp, fp, fn)

    def _extract_arguments(self, data):
        arguments = []

        for event in data.get("events", []):
            event_type = event.get("event_type")

            for argument in event.get("arguments", []):
                for mention in argument.get("mentions", []):
                    span = tuple(mention["span"])
                    arguments.append((span, event_type))

        return arguments