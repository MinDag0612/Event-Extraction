"""Optional inference ablation: discourage empty roots without banning them."""
import math
from transformers import LogitsProcessor


class EmptyRootPenalty(LogitsProcessor):
    def __init__(self, open_id, close_id, pad_id, penalty=0.):
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError('Empty-root penalty must be finite and nonnegative')
        self.open_id, self.close_id, self.pad_id = open_id, close_id, pad_id
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        if not self.penalty:
            return scores
        # mT5 starts decoding with PAD. Penalize only the FIRST root-closing
        # decision, never argument/event closures or EOS. Preserve -inf masks.
        rows = []
        for index, prefix in enumerate(input_ids.tolist()):
            if prefix and prefix[0] == self.pad_id:
                prefix = prefix[1:]
            if prefix == [self.open_id]:
                rows.append(index)
        if not rows:
            return scores
        adjusted = scores.clone()
        adjusted[rows, self.close_id] -= self.penalty
        return adjusted
