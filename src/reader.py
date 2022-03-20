from .entities import Dataset


class Reader:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def read(self, src_path: str, tgt_path: str, data_name: str):
        dataset = Dataset(data_name, self._tokenizer)

        with open(src_path, mode='r', encoding='utf-8-sig') as f:
            src_data = f.read().split('\n')
        if src_data[-1] == '':
            src_data.pop()

        with open(tgt_path, mode='r', encoding='utf-8-sig') as f:
            tgt_data = f.read().split('\n')
        if tgt_data[-1] == '':
            tgt_data.pop()

        assert len(src_data) == len(tgt_data)
        for (src_sent, tgt_sent) in zip(src_data, tgt_data):
            dataset.add_sample(src_sent.strip(), tgt_sent.strip())

        return dataset
