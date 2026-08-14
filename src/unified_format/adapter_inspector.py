import json
from pathlib import Path
from typing import Any


class AdapterInspector:
    def __init__(self, adapter, dataset_name):
        self.adapter = adapter
        self.dataset_name = dataset_name

    def inspect_sample(self, index: int = 0):
        """Show raw and unified values for one JSONL sample."""
        if index < 0:
            raise ValueError("index must be greater than or equal to 0")

        project_root = Path(__file__).resolve().parents[2]
        data_path = project_root / "data" / "raw" / self.dataset_name / "train.jsonl"

        with data_path.open("r", encoding="utf-8") as file:
            for current_index, line in enumerate(file):
                if current_index == index:
                    raw = json.loads(line)
                    break
            else:
                raise IndexError(
                    f"Sample {index} does not exist in {data_path.as_posix()}"
                )

        adapted = self.adapter.adapt(raw)
        tee, elbow, pipe = "\u251c\u2500\u2500", "\u2514\u2500\u2500", "\u2502   "

        print(f"Dataset: {self.dataset_name}")
        print(f"Sample: {index}\n")

        print("RAW")
        raw_fields = list(raw)
        for position, field_name in enumerate(raw_fields):
            branch = elbow if position == len(raw_fields) - 1 else tee
            value = json.dumps(raw[field_name], ensure_ascii=False)
            print(f"{branch} {field_name}: {value}")

        print("\nADAPTED")
        print(f"{tee} id: {adapted.id!r}")
        print(f"{tee} text: {adapted.text!r}")
        print(f"{elbow} events")

        events = getattr(adapted, "events", [])
        for event_index, event in enumerate(events):
            is_last_event = event_index == len(events) - 1
            event_branch = elbow if is_last_event else tee
            event_prefix = "    " if is_last_event else pipe
            print(f"    {event_branch} Event[{event_index}]")
            print(f"    {event_prefix}{tee} event_type: {event.event_type!r}")
            print(f"    {event_prefix}{tee} trigger: {event.trigger!r}")
            print(f"    {event_prefix}{elbow} arguments")

            arguments = getattr(event, "arguments", [])
            for argument_index, argument in enumerate(arguments):
                branch = elbow if argument_index == len(arguments) - 1 else tee
                mentions = json.dumps(argument.mentions, ensure_ascii=False)
                print(f"    {event_prefix}    {branch} {argument.role}: {mentions}")

    def inspect_structure(self, sample):
        """Show the structure of adapted sample."""
        pass

    def inspect_statistics(self, n: int = 100):
        """Analyze adapted samples."""
        pass

    def validate(self, sample):
        """Check unified-format constraints."""
        pass

    def compare(self, raw: Any, adapted: Any):
        """Show how raw fields were mapped to unified fields."""
        pass
