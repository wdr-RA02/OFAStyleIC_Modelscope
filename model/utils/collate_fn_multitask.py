import os

from modelscope.models.base import Model
from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.hub.snapshot_download import snapshot_download

from functools import partial

def build_collator(model_dir, rev):
    print("Build multi_task collator")
    # reluctantly init a tokenizer to get ids
    model = Model.from_pretrained(
            model_dir, revision=rev, invoked_by="trainer")
    # tokenizer=OFATokenizer.from_pretrained(model_dir)
    pad_id=model.tokenizer.pad_token_id
    eos_id=model.tokenizer.eos_token_id

    del model
    return partial(collate_mul_tasks, pad_index=pad_id, eos_index=eos_id)


def collate_mul_tasks(sample, pad_index, eos_index):
    '''
    将包含多个训练任务的sample降维为符合collate_fn要求的sample

    args:
    sample: [(task1_0, task2_0, ...), (task1_1,task2_1,...)]
    pad_index, eos_index: tokenizer的bos_id和eos_id
    
    output:
    sample_out=[task1_0, task2_0, ..., task1_1, task2_1, ...]
    '''
    len_items=map(len,sample)
    # check whether all samples in batch has equal num of tasks
    assert len(set(len_items))==1, \
        "The task numbers for each batch must be the same, got {}".format(list(len_items))
    # unsqueeze the task s.t.
    # sample=[task1_0, task2_0, ..., task1_1, task2_1, ...]
    sample_unsqueeze=[task for item_in_batch in sample for task in item_in_batch]

    return collate_fn(sample_unsqueeze, pad_idx=pad_index, eos_idx=eos_index)