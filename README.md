# kdprompt

```bash
python baseline.py --dataset cora
python train_teacher.py --dataset ogbn-arxiv --prompts_dim 128 --save_results --seed
python test_prompt.py --dataset cora --prompts_dim 128 --seed
```

different model hyper-parameters for student and prompt

------------

inductive learning: model parameters inherit from transparnt student?
(also feature_noise)

------------

TODO:

* fanout of ogbn-arxiv
* model.p require grad at pre-training stage
