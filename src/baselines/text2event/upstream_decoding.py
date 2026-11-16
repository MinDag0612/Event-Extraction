"""mT5 callback adaptation of Text2Event extraction/extract_constraint.py.

Reference revision dbe5a0ae933d639980c8e1f62c141511c12cefd6.
Follows upstream tree/treespan rules: label trie + contiguous SOURCE SUBWORDS,
empty root allowed, no repetition quota, no original-token boundary rule and no
reserved closing budget. max_new_tokens is enforced by generate; unfinished
outputs are scored as invalid. Unlike upstream interface, source IDs here have
already had the task prefix removed by the caller. Default adds ambiguous-label
prefix handling for the project schemas; resolve_label_prefixes=False reproduces
the historical label resolution for comparison.
"""
import json


class UpstreamConstraint:
    def __init__(self, tokenizer, schema_path, source_ids, flat=False, resolve_label_prefixes=True):
        self.t = tokenizer
        types, roles, _ = [json.loads(line) for line in schema_path.read_text().splitlines()]
        self.open = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.close = tokenizer.convert_tokens_to_ids('<extra_id_1>')
        self.flat = flat
        self.resolve_label_prefixes = resolve_label_prefixes
        self.type_tree = self.label_tree(types + roles if flat else types)
        self.role_tree = self.label_tree(roles)
        self.source = list(source_ids)
        if tokenizer.eos_token_id in self.source:
            self.source = self.source[:self.source.index(tokenizer.eos_token_id)]
        self.budget_constrained = False  # Interface compatibility; no such rule upstream.
        self.repetition_policy = 'upstream-compatible' if resolve_label_prefixes else 'upstream-exact'
        self.event_repeat_limit = self.argument_repeat_limit = None

    def label_tree(self, labels):
        tree = {}
        for label in labels:
            node = tree
            for token in self.t.encode(label, add_special_tokens=False):
                node = node.setdefault(token, {})
            node[None] = True
        return tree

    def source_next(self, generated, endings):
        if not generated:
            return self.source
        n = len(generated)
        following = [self.source[i+n] for i in range(len(self.source)-n)
                     if self.source[i:i+n] == generated]
        return following + endings

    def label_and_source(self, generated, tree, endings):
        if self.resolve_label_prefixes:
            # Keep BOTH interpretations when a label prefixes another label:
            # continue that longer label, or start a mention after the short one.
            # Required for PHEE Treatment/Treatment.Drug and mixed type/role labels.
            allowed=set();node=tree
            for index in range(len(generated)+1):
                if None in node:
                    span=generated[index:]
                    if not span:
                        allowed.update(self.source)
                    else:
                        n=len(span)
                        matches=[i for i in range(len(self.source)-n+1)
                                 if self.source[i:i+n]==span]
                        if matches:
                            allowed.update(endings)
                            allowed.update(self.source[i+n] for i in matches if i+n<len(self.source))
                if index==len(generated):
                    allowed.update(token for token in node if token is not None)
                    break
                if generated[index] not in node:
                    break
                node=node[generated[index]]
            if not allowed:
                raise ValueError('No legal label/source continuation')
            return sorted(allowed)
        # Exact historical resolution, retained for differential comparison.
        node = tree
        for index, token in enumerate(generated):
            node = node[token]
            if None in node:
                # Same resolution as upstream: once a complete label is found,
                # the remainder is treated as a source span.
                return self.source_next(generated[index+1:], endings)
        return [token for token in node if token is not None]

    def __call__(self, batch_id, prefix):
        ids = prefix.tolist()
        if ids[-1] == self.t.pad_token_id:
            return [self.open]
        special = [(i, t) for i, t in enumerate(ids) if t in {self.open, self.close}]
        if not special:
            return [self.t.eos_token_id]
        index, last = special[-1]
        depth = ids.count(self.open)-ids.count(self.close)
        if depth == 0:
            return [self.t.eos_token_id]
        if depth == 1:
            return [self.open, self.close]
        if depth == 2:
            if last == self.close:
                if self.flat:
                    raise ValueError('Invalid nested span tree')
                return [self.open, self.close]
            tree = self.type_tree
            endings = [self.close] if self.flat else [self.open, self.close]
        elif depth == 3 and not self.flat:
            tree, endings = self.role_tree, [self.close]
        else:
            return [self.t.eos_token_id]
        if ids[-1] == self.open:
            return list(tree)
        return self.label_and_source(ids[index+1:], tree, endings)
