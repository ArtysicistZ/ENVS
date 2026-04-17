import random
from collections import defaultdict

from ..protocol import DataProto, collate_fn


class ReplayBuffer():

    def __init__(self, json_path, buffer_size):
        self.buffer_size = buffer_size
        self.pos_dataset = defaultdict(lambda: defaultdict(list))

    def update_replay_buffer(self, task_config, batch_item, eval_result, replay_tag="clean_success"):
        task_id = task_config["task_id"]
        if eval_result <= 0.1:
            return

        task_replay_buffer = self.pos_dataset[task_id][replay_tag]
        task_replay_buffer.append(batch_item)

        if len(task_replay_buffer) > self.buffer_size:
            task_replay_buffer.pop(0)

    def update_replay_buffer_batch(self, task_configs, batch, replay_tags=None):
        eval_results = batch.batch["eval_results"].tolist()
        if replay_tags is None:
            replay_tags = ["clean_success"] * len(eval_results)

        for task_config, batch_item, eval_result, replay_tag in zip(task_configs, batch, eval_results, replay_tags):
            self.update_replay_buffer(task_config, batch_item, eval_result, replay_tag=replay_tag)

    def _get_candidates(self, task_id, preferred_tags=None):
        task_buffers = self.pos_dataset.get(task_id)
        if not task_buffers:
            return [], None

        if preferred_tags:
            for tag in preferred_tags:
                if task_buffers.get(tag):
                    return task_buffers[tag], tag

        merged = []
        for tag, items in task_buffers.items():
            merged.extend(items)
        return merged, None

    def get_pos(self, task_id, num_samples=1, preferred_tags=None):
        datalist, selected_tag = self._get_candidates(task_id, preferred_tags=preferred_tags)
        if not datalist:
            return DataProto()

        chosen = random.choices(datalist, k=num_samples)
        batch = collate_fn(chosen)
        if selected_tag is not None:
            batch.meta_info["replay_source_tag"] = selected_tag
        return batch
