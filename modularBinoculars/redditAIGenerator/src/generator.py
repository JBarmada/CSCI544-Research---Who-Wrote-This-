import random

from src import config
from src.prompting import PROMPT_VERSION, build_all_mode_prompts, prompt_template_id
from src.validation import cleanup_generated_text, length_bin, text_fingerprint, validate_generated_text, word_count


class RedditAIGenerator:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_generated_words: int,
        dry_run: bool,
        random_seed: int,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_generated_words = min_generated_words
        self.dry_run = dry_run
        self.random_seed = random_seed

        random.seed(random_seed)

        self.tokenizer = None
        self.model = None
        self.torch = None
        if not dry_run:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.torch.manual_seed(random_seed)

            print(f"Loading generator model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.torch.bfloat16 if self.torch.cuda.is_available() else self.torch.float32,
                device_map="auto",
            )
            self.model.eval()

    def _chat_prompts(self, seed_texts: list[str], mode: str) -> list[str]:
        prompts = []
        for seed_text in seed_texts:
            messages = build_all_mode_prompts(seed_text)[mode]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        return prompts

    def _dry_run_outputs(self, seed_texts: list[str], mode: str) -> list[str]:
        outputs = []
        for seed_text in seed_texts:
            base = cleanup_generated_text(seed_text)
            if mode == "controlled_rewrite":
                outputs.append(f"Dry run rewrite: {base}")
            elif mode == "continuation":
                outputs.append(f"{base} This follow-up continues the same discussion in a plausible way.")
            else:
                outputs.append(f"Dry run polished version: {base}")
        return outputs

    def _generate_mode_batch(self, seed_texts: list[str], mode: str) -> list[str]:
        if self.dry_run:
            return self._dry_run_outputs(seed_texts, mode)

        prompts = self._chat_prompts(seed_texts, mode)
        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = []
        prompt_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
        for row_ids, prompt_len in zip(output_ids, prompt_lengths):
            new_tokens = row_ids[int(prompt_len):]
            generated.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return generated

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _row_from_generation(
        self,
        record,
        generated_text: str,
        mode: str,
        generated_from_split: str,
    ) -> dict:
        wc = word_count(generated_text)
        return {
            "text": generated_text,
            "source_text": str(record["text"]),
            "source_id": str(record["id"]),
            "subreddit": record["subreddit"],
            "domain": record["domain"],
            "post_type": record["post_type"],
            "year": self._safe_int(record["year"]),
            "word_count": wc,
            "length_bin": length_bin(wc),
            "score": self._safe_int(record["score"]),
            "created_utc": self._safe_int(record["created_utc"]),
            "generation_mode": mode,
            "model_name": self.model_name,
            "prompt_template": prompt_template_id(mode),
            "prompt_version": PROMPT_VERSION,
            "generated_from_split": generated_from_split,
            "is_ai_generated": True,
        }

    def generate_rows(self, df, generated_from_split, checkpoint, checkpoint_every: int) -> list[dict]:
        all_rows = list(checkpoint.rows)
        completed_since_save = 0
        seen_fingerprints = set(checkpoint.seen_fingerprints)

        pending = df[~df["id"].astype(str).isin(checkpoint.completed_ids)].reset_index(drop=True)
        total_pending = len(pending)
        print(f"Pending seeds to generate: {total_pending:,}")

        for batch_start in range(0, total_pending, self.batch_size):
            batch_df = pending.iloc[batch_start:batch_start + self.batch_size]
            records = batch_df.to_dict(orient="records")
            seed_texts = [str(record["text"]) for record in records]

            mode_outputs = {}
            for mode in config.GENERATION_MODES:
                mode_outputs[mode] = self._generate_mode_batch(seed_texts, mode)

            for idx, record in enumerate(records):
                source_rows = []
                source_fingerprints = []

                for mode in config.GENERATION_MODES:
                    ok, cleaned = validate_generated_text(
                        generated_text=mode_outputs[mode][idx],
                        source_text=str(record["text"]),
                        min_generated_words=self.min_generated_words,
                        seen_fingerprints=seen_fingerprints,
                    )
                    if not ok:
                        continue

                    source_rows.append(
                        self._row_from_generation(
                            record=record,
                            generated_text=cleaned,
                            mode=mode,
                            generated_from_split=generated_from_split,
                        )
                    )
                    source_fingerprints.append(text_fingerprint(cleaned))

                checkpoint.add_seed_rows(str(record["id"]), source_rows, source_fingerprints)
                all_rows.extend(source_rows)
                completed_since_save += 1

                if completed_since_save >= checkpoint_every:
                    checkpoint.save()
                    completed_since_save = 0

            processed = min(batch_start + len(records), total_pending)
            print(f"Processed seeds: {processed:,} / {total_pending:,}")

        checkpoint.save()
        return all_rows
