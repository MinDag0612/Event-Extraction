"""Constraint cho bản Text2Event-mT5 adaptation (không phải decoder gốc).

Rules: đúng cây/schema; mention là đoạn token gốc liên tiếp; chừa đủ chỗ đóng
cây và EOS. repetition_policy='source-occurrence' còn giới hạn số lần sinh
(type, trigger surface) trong câu và (role, argument surface) trong một event
bằng số span đầu vào có cùng chuỗi token hóa nhân repeat_limit. Repeat limit
lấy từ TRAIN để không cấm cấu trúc lặp có thật trong annotation. Không dùng gold khi inference.

Vẫn cho phép cây rỗng, event không argument, nhiều argument khác nhau cùng role,
cùng surface ở nhiều vị trí và dùng lại argument trong các event khác nhau.
Không đảm bảo đúng ngữ nghĩa; mapping type→roles phụ thuộc schema được cung cấp.
Policy 'none' giữ rule cũ để so sánh. Greedy/beam được chọn ở evaluation.py.
"""
import json
from collections import Counter
from dataclasses import dataclass, replace


class ConstraintDeadEnd(ValueError):
    pass


@dataclass(frozen=True)
class State:
    phase: str = 'start'
    kind: str | None = None
    value: tuple | int | None = None
    role: str | None = None
    events: tuple = ()
    arguments: tuple = ()


def increment(counts, label, node):
    values = {(name, idx): count for name, idx, count in counts}
    values[label, node] = values.get((label, node), 0) + 1
    return tuple(sorted((name, idx, count) for (name, idx), count in values.items()))


class TreeConstraint:
    def __init__(self, tokenizer, schema_path, input_ids, original_tokens=None,
                 max_new_tokens=704, repetition_policy='source-occurrence',
                 event_repeat_limit=1, argument_repeat_limit=1, flat=False, flat_repeat_limit=None, allow_empty_tree=True):
        if repetition_policy not in {'source-occurrence', 'none'}:
            raise ValueError('Unknown repetition policy')
        if max_new_tokens < 3:
            raise ValueError('Need at least root open, root close and EOS')
        if any(type(v) is not int or v < 1 for v in (event_repeat_limit, argument_repeat_limit)):
            raise ValueError('Repeat limits must be positive integers')
        self.event_repeat_limit = event_repeat_limit
        self.argument_repeat_limit = argument_repeat_limit
        self.allow_empty_tree = allow_empty_tree
        self.flat = flat
        self.flat_repeat_limit = flat_repeat_limit if flat_repeat_limit is not None else max(event_repeat_limit, argument_repeat_limit)
        if type(self.flat_repeat_limit) is not int or self.flat_repeat_limit < 1:
            raise ValueError('Flat repeat limit must be a positive integer')
        self.repetition_policy = repetition_policy
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
        self.eos, self.pad = tokenizer.eos_token_id, tokenizer.pad_token_id
        self.labels = {label: tuple(tokenizer.encode(label, add_special_tokens=False))
                       for label in set(types+roles)}
        if any(not seq for seq in self.labels.values()) or len(set(self.labels.values())) != len(self.labels):
            raise ValueError('Tokenizer collapses different schema labels')
        if original_tokens is None:
            raise ValueError('Original tokens required for surface constraints')
        # Đếm span, không chỉ surface duy nhất: hai mention giống mặt chữ ở hai
        # vị trí vẫn được sinh hai lần. input_ids giữ để tương thích giao diện.
        surfaces = Counter(' '.join(original_tokens[a:b])
                           for a in range(len(original_tokens))
                           for b in range(a+1, len(original_tokens)+1))
        self.trie = [{}]
        self.capacity = Counter()
        for surface, count in surfaces.items():
            seq = tokenizer.encode(surface, add_special_tokens=False)
            if not seq or any(i in {self.o, self.c, self.eos, self.pad} for i in seq):
                raise ValueError('Unrepresentable input surface')
            node = 0
            for token in seq:
                if token not in self.trie[node]:
                    self.trie[node][token] = len(self.trie)
                    self.trie.append({})
                node = self.trie[node][token]
            self.capacity[node] += count
        self.terminal = set(self.capacity)
        self.types = tuple(dict.fromkeys(types + roles)) if flat else tuple(types)
        self.roles = set(roles)
        self.max_new_tokens = max_new_tokens
        self.budget_constrained = False
        self._prefix, self._states = (), {State()}
        self._distance_cache = {}

    def repeat_limit(self, label, trigger=True):
        # Flat curriculum stores role–argument leaves at the same depth as
        # type–trigger leaves. Preserve TRAIN-derived role multiplicity there.
        if self.flat:
            return self.flat_repeat_limit
        return self.event_repeat_limit if trigger else self.argument_repeat_limit

    def distances(self, label, counts, limit=1):
        """Khoảng cách tới terminal chưa hết quota; cắt cả nhánh đã dùng hết.

        Không chỉ cấm token đóng tại terminal trùng: cách đó có thể khiến model
        đi vào nhánh không còn đường ra, gây ConstraintDeadEnd.
        """
        key = (label, tuple(item for item in counts if item[0] == label)
               if self.repetition_policy != 'none' else (), limit)
        if key not in self._distance_cache:
            used = {node: count for name, node, count in key[1] if name == label}
            distance = [float('inf')] * len(self.trie)
            for node in range(len(self.trie)-1, -1, -1):
                available = node in self.capacity and (
                    self.repetition_policy == 'none' or used.get(node, 0) < self.capacity[node] * limit)
                distance[node] = 0 if available else min(
                    (1+distance[child] for child in self.trie[node].values()), default=float('inf'))
            self._distance_cache[key] = distance
        return self._distance_cache[key]

    def choices(self, state):
        labels = self.types if state.phase == 'type' else self.mapping[state.kind]
        counts = state.events if state.phase == 'type' else state.arguments
        return [label for label in labels if self.distances(label, counts, self.repeat_limit(label, state.phase == 'type'))[0] != float('inf')]

    def transitions(self, state):
        phase = state.phase
        def move(**kwargs):
            return replace(state, **kwargs)
        def add(out, token, following):
            out.setdefault(token, set()).add(following)
        if phase == 'start':
            return {self.o: {move(phase='root')}}
        if phase == 'root':
            # Khi cấm cây rỗng, chỉ đóng root sau ít nhất một leaf/event.
            # Không đọc gold; áp dụng cùng quy tắc cho mọi câu trong lượt thử.
            out = {self.c: {move(phase='done')}} if self.allow_empty_tree or state.events else {}
            following = move(phase='type', kind=None, value=(), role=None, arguments=())
            if self.choices(following):
                out[self.o] = {following}
            return out
        if phase in {'done', 'ended'}:
            return {self.eos: {move(phase='ended')}}
        if phase in {'type', 'role'}:
            out = {}
            for label in self.choices(state):
                seq = self.labels[label]
                if seq[:len(state.value)] != state.value:
                    continue
                if len(state.value) < len(seq):
                    token = seq[len(state.value)]
                    add(out, token, move(value=state.value+(token,)))
                else:
                    counts = state.events if phase == 'type' else state.arguments
                    distances = self.distances(label, counts, self.repeat_limit(label, phase == 'type'))
                    for token, node in self.trie[0].items():
                        if distances[node] != float('inf'):
                            add(out, token, move(phase='trigger' if phase == 'type' else 'argument',
                                kind=label if phase == 'type' else state.kind,
                                role=label if phase == 'role' else None, value=node))
            return out
        if phase in {'trigger', 'argument'}:
            label = state.kind if phase == 'trigger' else state.role
            counts = state.events if phase == 'trigger' else state.arguments
            distances = self.distances(label, counts, self.repeat_limit(label, phase == 'trigger'))
            out = {}
            for token, node in self.trie[state.value].items():
                if distances[node] != float('inf'):
                    add(out, token, move(value=node))
            if distances[state.value] == 0:
                if phase == 'trigger':
                    events = increment(state.events, state.kind, state.value)
                    add(out, self.c, move(phase='root', kind=None, value=None, events=events, arguments=()))
                    following = move(phase='role', value=(), events=events, arguments=())
                    if not self.flat and self.choices(following):
                        add(out, self.o, following)
                else:
                    arguments = increment(state.arguments, state.role, state.value)
                    add(out, self.c, move(phase='after_argument', value=None, role=None, arguments=arguments))
            return out
        if phase == 'after_argument':
            # Quota argument chỉ trong event hiện tại; event tiếp theo được dùng lại.
            out = {self.c: {move(phase='root', kind=None, value=None, arguments=())}}
            following = move(phase='role', value=())
            if self.choices(following):
                out[self.o] = {following}
            return out
        raise ConstraintDeadEnd(f'Unknown state {state}')

    def minimum_completion(self, state):
        # Ngân sách phải tính cả leaf bắt buộc, không chỉ root close + EOS.
        if not self.allow_empty_tree and not state.events:
            if state.phase == 'start':
                return 1 + self.minimum_completion(replace(state, phase='root'))
            if state.phase == 'root':
                return 1 + self.minimum_completion(replace(state, phase='type', kind=None, value=(), role=None, arguments=()))
        fixed = {'start': 3, 'root': 2, 'done': 1, 'ended': 0, 'after_argument': 3}
        if state.phase in fixed:
            return fixed[state.phase]
        trigger = state.phase in {'trigger', 'type'}
        counts = state.events if trigger else state.arguments
        closing = 3 if trigger else 4
        if state.phase in {'trigger', 'argument'}:
            label = state.kind if trigger else state.role
            return self.distances(label, counts, self.repeat_limit(label, trigger))[state.value]+closing
        return min((len(self.labels[label])-len(state.value)+self.distances(label, counts, self.repeat_limit(label, trigger))[0]+closing
                    for label in self.choices(state)
                    if self.labels[label][:len(state.value)] == state.value), default=float('inf'))

    def __call__(self, batch_id, prefix):
        # Callback của generate(): grammar/schema/source/quota + ngân sách đóng cây.
        ids = prefix.tolist()
        if ids and ids[0] == self.pad:
            ids = ids[1:]
        key = tuple(ids)
        if key[:len(self._prefix)] == self._prefix:
            states, remaining = self._states, key[len(self._prefix):]
        else:
            states, remaining = {State()}, key
        for token in remaining:
            following = set()
            for state in states:
                following.update(self.transitions(state).get(token, set()))
            states = following
            if not states:
                raise ConstraintDeadEnd(f'Invalid prefix: position={len(key)}, last_token={token}, flat={self.flat}')
        self._prefix, self._states = key, states
        allowed = set()
        budget = self.max_new_tokens-len(key)
        for state in states:
            for token, following in self.transitions(state).items():
                if any(self.minimum_completion(next_state) <= budget-1 for next_state in following):
                    allowed.add(token)
                else:
                    self.budget_constrained = True
        if not allowed:
            raise ConstraintDeadEnd(f'No legal next token: position={len(key)}, budget={budget}, phases={sorted({s.phase for s in states})}, flat={self.flat}')
        return sorted(allowed)
