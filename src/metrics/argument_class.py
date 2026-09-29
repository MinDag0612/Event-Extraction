from src.metrics.base_metrics import BaseMetrics


class AC(BaseMetrics):
    """
    Argument Classification

    An argument is correct if:
    - argument span matches
    - argument role matches
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
            for argument in event.get("arguments", []):
                role = argument.get("role")

                for mention in argument.get("mentions", []):
                    span = tuple(mention["span"])

                    arguments.append((span, role))

        return arguments