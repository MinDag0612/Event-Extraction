from src.metrics.trigger_ident import TI
from src.metrics.trigger_class import TC
from src.metrics.argument_ident import AI
from src.metrics.argument_class import AC


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
            raise ValueError("predictions and golds must have the same length")

        # Reset metrics để mỗi lần gọi là một lượt đánh giá độc lập
        self.__init__()

        for prediction, gold in zip(predictions, golds):
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