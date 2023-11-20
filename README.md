# kdprompt

```bash
python autoencoder.py --dropout_autoencoder 0.7 --dropout_MLP 0.3 --lamb 0.5
python train_teacher.py --dataset ogbn-arxiv --prompts_dim 64 --num_exp 5
python train_student.py --dataset ogbn-arxiv --student MLP --prompts_dim 64 --save_results --feature_aug_k 1 --upstream_feature_aug_k 0 --lamb .5 --num_exp 5  # default lamb=0 (aka all GNN)
python test_prompt.py --dataset cora --student MLP --prompts_dim 64 --feature_aug_k 1 --upstream_feature_aug_k 1 --num_exp 5
```

```bash
# -prompt
python train_teacher.py --dataset ogbn-arxiv --teacher MLP --prompts_dim 64 --save_results --feature_aug_k 1 --num_exp 5
python test_prompt.py --dataset cora --teacher MLP --prompts_dim 64 --feature_aug_k 1 --upstream_feature_aug_k 1 --num_exp 5
# -aug
python train_student.py --dataset ogbn-arxiv --student MLP --prompts_dim 64 --save_results --lamb .5 --num_exp 5
python test_prompt.py --dataset cora --student MLP --prompts_dim 64 --num_exp 5
# -both
python train_teacher.py --dataset ogbn-arxiv --teacher MLP --prompts_dim 64 --save_results --num_exp 5
python test_prompt.py --dataset cora --teacher MLP --prompts_dim 64 --num_exp 5
```

```bash
# Baseline
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora --feature_aug_k 0 --num_exp 5
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora --feature_aug_k 1 --num_exp 5
python train_teacher.py --exp_setting tran --teacher MLP --dataset cora --feature_aug_k 0 --num_exp 5
python train_teacher.py --exp_setting tran --teacher MLP --dataset cora --feature_aug_k 1 --num_exp 5
```

different model hyper-parameters for student and prompt

------------

inductive learning: model parameters inherit from transparnt student?
(also feature_noise)
