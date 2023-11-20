# kdprompt

```bash
python autoencoder.py --dataset cora
python train_teacher.py --dataset ogbn-arxiv --prompts_dim 64 --save_results --seed
python test_prompt.py --dataset cora --prompts_dim 64 --seed
```

different model hyper-parameters for student and prompt

------------

inductive learning: model parameters inherit from transparnt student?
(also feature_noise)

------------

TODO:

* implement baseline
* fanout of ogbn-arxiv
* planetoid dataset
* model.p require grad at pre-training stage
* loss function: CrossEntropy, CosineSimilarity, etc.
* normalize logits instead of prompt (diffrenet normalizing functions)
