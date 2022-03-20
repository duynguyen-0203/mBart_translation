from datasets import load_metric

import torch


class Evaluator:
    def __init__(self, dataset, tokenizer):
        self._metric = load_metric('bleu')
        self._dataset = dataset
        self._tokenizer = tokenizer

        self.predictions = []
        self.references = []

        self._convert_references()

    def eval_batch(self, outputs: torch.tensor):
        for output in outputs:
            self.predictions.append(self._tokenizer.convert_ids_to_tokens(output, skip_special_tokens=True))

    def compute_scores(self):
        assert len(self.predictions) == len(self.references)
        results = self._metric.compute(predictions=self.predictions, references=self.references)

        return results['bleu']

    def _convert_references(self):
        for sample in self._dataset.samples:
            self.references.append(self._tokenizer.tokenize(sample.target))
