"""Tree grammar with complete original-token surfaces and a reserved closing budget."""
import json


class ConstraintDeadEnd(ValueError):
    pass


class TreeConstraint:
    def __init__(self, tokenizer, schema_path, input_ids, original_tokens=None, max_new_tokens=704):
        self.t = tokenizer
        lines = [json.loads(x) for x in schema_path.read_text().splitlines()]
        if len(lines) != 3:
            raise ValueError('event.schema must contain exactly three lines')
        types, roles, mapping = lines
        if set(types) != set(mapping) or any(set(rs)-set(roles) for rs in mapping.values()):
            raise ValueError('Inconsistent schema')
        self.mapping = mapping
        self.o = tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.c = tokenizer.convert_tokens_to_ids('<extra_id_1>')
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.labels = {label: tuple(tokenizer.encode(label, add_special_tokens=False)) for label in set(types+roles)}
        if len(set(self.labels.values())) != len(self.labels):
            raise ValueError('Tokenizer collapses different schema labels')
        # Encode complete original-token spans, never arbitrary source subword suffixes.
        if original_tokens is None:
            raise ValueError('Original tokens required for surface constraints')
        surfaces = list(dict.fromkeys(' '.join(original_tokens[a:b])
                       for a in range(len(original_tokens)) for b in range(a+1,len(original_tokens)+1)))
        self.trie = [{}]
        self.terminal = set()
        self.surface_min = [10**9]
        for surface in surfaces:
            seq = tokenizer.encode(surface, add_special_tokens=False)
            if not seq or any(i in {self.o,self.c,self.eos,self.pad} for i in seq):
                raise ValueError('Unrepresentable input surface')
            node = 0
            for token in seq:
                if token not in self.trie[node]:
                    self.trie[node][token] = len(self.trie)
                    self.trie.append({});self.surface_min.append(10**9)
                node = self.trie[node][token]
            self.terminal.add(node)
        for node in range(len(self.trie)-1,-1,-1):
            self.surface_min[node] = 0 if node in self.terminal else min(
                (1+self.surface_min[v] for v in self.trie[node].values()),default=10**9)
        self.max_new_tokens = max_new_tokens
        self.budget_constrained = False
        self.types = tuple(types)
        self._prefix = ()
        self._states = {('start',None,None)}

    def transitions(self, state):
        phase, kind, value = state
        if phase == 'start': return {self.o: {('root',None,None)}}
        if phase == 'root': return {self.o: {('type',None,())}, self.c: {('done',None,None)}}
        if phase == 'done': return {self.eos: {('ended',None,None)}}
        if phase == 'ended': return {self.eos: {state}}
        if phase in {'type','role'}:
            choices = self.types if phase == 'type' else self.mapping[kind]
            out = {}
            for label in choices:
                seq = self.labels[label]
                if seq[:len(value)] != value: continue
                if len(value) < len(seq):
                    out.setdefault(seq[len(value)],set()).add((phase,kind,value+(seq[len(value)],)))
                else:
                    next_kind = label if phase == 'type' else kind
                    for token, node in self.trie[0].items():
                        out.setdefault(token,set()).add(('trigger' if phase=='type' else 'argument', next_kind, node))
            return out
        if phase in {'trigger','argument'}:
            out = {}
            for token, node in self.trie[value].items():
                out.setdefault(token,set()).add((phase,kind,node))
            if value in self.terminal:
                out.setdefault(self.c,set()).add(('root',None,None) if phase=='trigger' else ('after_argument',kind,None))
                if phase=='trigger' and self.mapping[kind]:
                    out.setdefault(self.o,set()).add(('role',kind,()))
            return out
        if phase == 'after_argument':
            return {self.o:{('role',kind,())},self.c:{('root',None,None)}}
        raise ConstraintDeadEnd(f'Unknown state {state}')

    def minimum_completion(self, state):
        phase,kind,value=state
        fixed={'start':3,'root':2,'done':1,'ended':0,'after_argument':3}
        if phase in fixed:return fixed[phase]
        if phase in {'trigger','argument'}:
            return self.surface_min[value]+(3 if phase=='trigger' else 4)
        choices=self.types if phase=='type' else self.mapping[kind]
        remaining=[len(self.labels[label])-len(value) for label in choices
                   if self.labels[label][:len(value)]==value]
        return min(remaining,default=10**9)+self.surface_min[0]+(3 if phase=='type' else 4)

    def __call__(self, batch_id, prefix):
        ids = prefix.tolist()
        if ids and ids[0] == self.pad: ids=ids[1:]  # mT5 decoder start
        key = tuple(ids)
        if key[:len(self._prefix)] == self._prefix:
            states = self._states
            remaining = key[len(self._prefix):]
        else:
            states = {('start',None,None)}
            remaining = key
        for token in remaining:
            following=set()
            for state in states: following.update(self.transitions(state).get(token,set()))
            states=following
            if not states: raise ConstraintDeadEnd('Generated prefix violates tree/schema/source constraint')
        self._prefix, self._states = key, states
        allowed=set()
        budget=self.max_new_tokens-len(key)
        for state in states:
            for token, following in self.transitions(state).items():
                if any(self.minimum_completion(next_state)<=budget-1 for next_state in following):
                    allowed.add(token)
                else:
                    self.budget_constrained=True
        if not allowed: raise ConstraintDeadEnd('No legal next token')
        return sorted(allowed)
