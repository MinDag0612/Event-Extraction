from src.metrics.trigger_ident import TI
from src.metrics.trigger_class import TC
from src.metrics.argument_ident import AI
from src.metrics.argument_class import AC

import json


import json

def parse_prediction(text):
    try:
        data = json.loads(text)

        if not isinstance(data, dict):
            return {"events": []}

        events = data.get("events")
        if not isinstance(events, list):
            return {"events": []}

        for event in events:
            if not isinstance(event, dict):
                return {"events": []}

            if not isinstance(event.get("event_type"), str):
                return {"events": []}

            triggers = event.get("trigger")
            if not isinstance(triggers, list) or not triggers:
                return {"events": []}

            for trigger in triggers:
                if not isinstance(trigger, dict):
                    return {"events": []}

                if not isinstance(trigger.get("text"), str):
                    return {"events": []}

                span = trigger.get("span")
                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or not all(type(x) is int for x in span)
                ):
                    return {"events": []}

            arguments = event.get("arguments")
            if not isinstance(arguments, list):
                return {"events": []}

            for argument in arguments:
                if not isinstance(argument, dict):
                    return {"events": []}

                if not isinstance(argument.get("role"), str):
                    return {"events": []}

                mentions = argument.get("mentions")
                if not isinstance(mentions, list):
                    return {"events": []}

                for mention in mentions:
                    if not isinstance(mention, dict):
                        return {"events": []}

                    if not isinstance(mention.get("text"), str):
                        return {"events": []}

                    span = mention.get("span")
                    if (
                        not isinstance(span, list)
                        or len(span) != 2
                        or not all(type(x) is int for x in span)
                    ):
                        return {"events": []}

        return data

    except (json.JSONDecodeError, TypeError):
        return {"events": []}
class Evaluator:
    def __init__(self):
        self.ti = TI()
        self.tc = TC()
        self.ai = AI()
        self.ac = AC()

    def evaluate_on_sample(self, prediction, gold):
        self.ti.update(prediction, gold)
        self.tc.update(prediction, gold)
        self.ai.update(prediction, gold)
        self.ac.update(prediction, gold)

        return {
            "TI": self.ti.compute(),
            "TC": self.tc.compute(),
            "AI": self.ai.compute(),
            "AC": self.ac.compute(),
        }
        
    def evaluate_on_file(self, path):
        pass
    


    def evaluate_on_all_sample(self, predictions, golds):
        if len(predictions) != len(golds):
            raise ValueError(
                "predictions and golds must have the same length"
            )

        self.__init__()

        for prediction, gold in zip(predictions, golds):
            prediction = parse_prediction(prediction)

            self.ti.update(prediction, gold)
            self.tc.update(prediction, gold)
            self.ai.update(prediction, gold)
            self.ac.update(prediction, gold)

        results = {
            "TI": self.ti.compute(),
            "TC": self.tc.compute(),
            "AI": self.ai.compute(),
            "AC": self.ac.compute(),
        }

        results["avg_f1"] = sum(
            results[name]["f1"]
            for name in ("TI", "TC", "AI", "AC")
        ) / 4

        return results