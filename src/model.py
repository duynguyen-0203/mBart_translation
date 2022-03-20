import torch
from transformers import MBartConfig, MBartForConditionalGeneration, MBartPreTrainedModel


class MBartTranslation(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig):
        super().__init__(config)
        self.mbart = MBartForConditionalGeneration(config)

        self.init_weights()

    def forward(self, encodings: torch.tensor, attention_masks: torch.tensor, labels: torch.tensor):
        outputs = self.mbart(input_ids=encodings, attention_mask=attention_masks, labels=labels)

        return outputs['logits']
