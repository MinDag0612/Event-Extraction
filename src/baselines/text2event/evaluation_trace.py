"""Read-only evaluation telemetry; no token mask, score or RNG changes."""
import json
import time


class EvaluationTrace:
    def __init__(self, path, callback_every=16, stream=True):
        self.path = path
        self.callback_every = callback_every
        self.stream = stream
        if callback_every < 1:
            raise ValueError('callback_every must be positive')
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **fields):
        row = {'event': event, **fields}
        line = json.dumps(row, ensure_ascii=False)
        # Reopen/flush each record so earlier samples survive a runtime loss.
        with self.path.open('a') as handle:
            handle.write(line + '\n')
        if self.stream:
            # Raw text is retained in the JSONL artifact, not repeated on stdout.
            compact = {k: v for k, v in row.items() if k != 'diagnostic'}
            print(json.dumps(compact, ensure_ascii=False), flush=True)


class SampleTimer:
    def __init__(self, observer, sample_id, index, total):
        self.observer = observer
        self.sample_id = sample_id
        self.began = time.perf_counter()
        self.active = {}
        self.seconds = {}
        self.callback_calls = 0
        self.callback_seconds = 0.
        self.emit('SAMPLE_START', index=index, total=total)

    def emit(self, event, **fields):
        if self.observer:
            self.observer.emit(event, id=self.sample_id, **fields)

    def start(self, phase):
        self.active[phase] = time.perf_counter()
        self.emit('PHASE_START', phase=phase)

    def end(self, phase):
        if phase in self.active:
            self.seconds[phase] = time.perf_counter() - self.active.pop(phase)
            self.emit('PHASE_END', phase=phase, seconds=self.seconds[phase])

    def wrap(self, constraint):
        if not self.observer or not callable(constraint):
            return constraint if callable(constraint) else None

        def observed(batch_id, prefix):
            self.callback_calls += 1
            traced = self.callback_calls == 1 or self.callback_calls % self.observer.callback_every == 0
            if traced:
                self.emit('CALLBACK_START', call=self.callback_calls, prefix_tokens=len(prefix))
            began = time.perf_counter()
            try:
                allowed = constraint(batch_id, prefix)
            except Exception as exc:
                self.emit('CALLBACK_ERROR', call=self.callback_calls, error_type=type(exc).__name__, reason=str(exc))
                raise
            finally:
                elapsed = time.perf_counter() - began
                self.callback_seconds += elapsed
            if traced:
                self.emit('CALLBACK_END', call=self.callback_calls, seconds=elapsed, allowed_tokens=len(allowed))
            return allowed
        return observed

    def finish(self, issue):
        if not self.observer:
            return
        self.end('generate')
        timing = {**self.seconds, 'callback_seconds': self.callback_seconds,
                  'callback_calls': self.callback_calls,
                  'sample_seconds': time.perf_counter() - self.began}
        issue['timing'] = timing
        self.emit('SAMPLE_RESULT', status=issue['status'], generated_tokens=issue.get('generated_tokens'),
                  timing=timing, diagnostic=issue)
