from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
import torch


class GenerationTrainer(Trainer):
    def __init__(
        self,
        generation_dataset,
        tokenizer,
        evaluator,
        max_new_tokens=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.generation_dataset = generation_dataset
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.max_new_tokens = max_new_tokens

    def evaluate_generation(self):
        predictions = []
        golds = []

        was_training = self.model.training
        old_use_cache = self.model.config.use_cache

        self.model.eval()
        self.model.config.use_cache = True

        try:
            with torch.inference_mode():
                for example in self.generation_dataset:
                    # Chỉ đưa input_text vào model
                    inputs = self.tokenizer(
                        example["input_text"],
                        return_tensors="pt",
                        add_special_tokens=True,
                    )

                    inputs = {
                        key: value.to(self.model.device)
                        for key, value in inputs.items()
                    }

                    input_length = inputs["input_ids"].shape[1]

                    # Model tự sinh output
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=(
                            self.tokenizer.pad_token_id
                            or self.tokenizer.eos_token_id
                        ),
                    )

                    # Chỉ lấy phần model vừa sinh, bỏ prompt
                    generated_tokens = generated_ids[:, input_length:]

                    prediction = self.tokenizer.decode(
                        generated_tokens[0],
                        skip_special_tokens=True,
                    )

                    predictions.append(prediction)

                    # Gold được lấy riêng từ dataset
                    golds.append({
                        "events": example["events"]
                    })

        finally:
            self.model.config.use_cache = old_use_cache

            if was_training:
                self.model.train()

        return self.evaluator.evaluate_on_all_sample(
            predictions,
            golds,
        )

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Chạy evaluation mặc định của Trainer
        # để tính validation loss
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Chạy Generation evaluation
        results = self.evaluate_generation()

        metrics = dict(output.metrics)

        # Trả metric dạng scalar để Trainer sử dụng
        # chọn best checkpoint và Early Stopping
        metrics[f"{metric_key_prefix}_avg_f1"] = results["avg_f1"]

        for name in ("TI", "TC", "AI", "AC"):
            metrics[f"{metric_key_prefix}_{name}_f1"] = (
                results[name]["f1"]
            )

        # Lưu kết quả chi tiết để xem sau
        self.generation_results = results

        return EvalLoopOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=metrics,
            num_samples=output.num_samples,
        )
