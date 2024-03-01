# Prompt for Transfer Learning

The goal of the Prompt for Transfer Learning framework is to effectively leverage the knowledge encapsulated in a graph neural network (GNN) model pretrained on a large dataset and to apply this knowledge to a smaller, downstream dataset for the task of node classification.

## Challenge

The framework aims to achieve the goal while addressing two fundamental challenges: feature space mismatch and label space mismatch between the pretraining and downstream datasets.

Feature space mismatch occurs when the features of the nodes in the pretraining dataset differ in dimensionality, scale, or distribution from those in the downstream dataset. This discrepancy hinders the direct application of a pretrained model to the downstream task because the model’s input layer and subsequent feature processing layers are tailored to the pretraining dataset's feature space.

Label space mismatch refers to the difference in the number and nature of the labels between the pretraining and downstream datasets. The pretrained model is originally trained to predict labels that may not correspond to those in the downstream task, which presents a challenge for direct model transfer.

## Model

The proposed framework, Prompt for Transfer Learning, attempts to address the node classification problem by adapting transfer learning paradigms to GNNs. It leverages the underlying similarities in structure and semantics between large pretraining datasets and smaller, domain-specific downstream datasets to facilitate knowledge transfer. By employing a pretraining-prompt architecture, the framework aims to tackle the challenges of feature space mismatch and label space mismatch, thus enabling the application of a single model across diverse classification tasks. The harmonization of feature vectors and the unification of label spaces are achieved through a graph autoencoder and a similarity-based classification approach, respectively. In doing so, we maintain the integrity of the pretrained model while optimizing a learnable prompt to adapt the pretrained knowledge to the downstream context.

### Addressing Feature Space Mismatch

The strategy for mitigating feature space mismatch involves two key steps: dimensionality alignment through a graph autoencoder and feature space harmonization using a learnable prompt.

* Dimensionality Alignment: A graph autoencoder is employed to preprocess the downstream dataset, reshaping its node feature vectors to match the dimensionality of the pretraining dataset.
* Feature Space Harmonization: A learnable tensor, or prompt $p$, is introduced to fine-tune the downstream feature space to align with the pretraining space.

#### Graph Autoencoder Preprocessing

The graph autoencoder serves as the first step in transforming the downstream dataset's node features to have the same dimensionality as the pretraining dataset. This unsupervised learning technique is adept at capturing the latent structure of graph data and projecting it into a consistent feature space suitable for downstream tasks.

#### Prompt-based Feature Space Transformation

Following the dimensionality alignment, the prompt $p$ is utilized to adjust the downstream feature vectors. This prompt is a learnable tensor that, when element-wise multiplied with the node feature vectors, serves as a bridge between the pretraining and downstream feature spaces. The prompt is optimized during training to minimize the discrepancy between the two feature spaces, thus enhancing the model's ability to generalize across different datasets.

### Addressing Label Space Mismatch

To reconcile the differences in label spaces between the pretraining and downstream datasets, the classification problem is reframed as a similarity measurement task:

* Class-Prototype Calculation: For each class in the downstream dataset, compute a class-prototype by averaging the embeddings of all nodes within that class.
* Similarity-based Node Classification: Assign nodes to the class whose prototype has the highest cosine similarity to the node's embedding.

#### Class-Prototype Conceptualization

The class-prototype is a central component of the framework, representing the collective embedding of all nodes within a class. It acts as a reference point for classification, encapsulating the essential features that characterize each class in the downstream dataset.

#### Cosine Similarity and Cross-Entropy Loss

Using the cosine similarity measure, we compare the embedding of a querying node with all class-prototypes. The class with the highest similarity score is predicted as the node's class. The similarity scores are then integrated into a cross-entropy loss function, which guides the learning process for the prompt $p$.

### Model Training

During the downstream task training, the GNN (either GraphSAGE or GAT) parameters remain static, preserving the knowledge from pretraining. The prompt $p$ emerges as the sole learnable parameter, ensuring that the adaptation to the downstream task is precise and focused.

## Experiments

### Setup Environment

```bash
conda create -n kdprompt python=3.10
conda activate kdprompt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/cu118 dgl
conda install -c conda-forge ogb
conda install pyg -c pyg
conda install category_encoders pyyaml
```

### Prepare Dataset

```bash
python autoencoder.py --dataset cora
python autoencoder.py --dataset citeseer
python autoencoder.py --dataset pubmed
```

### Run

```bash
python baseline.py --model SAGE --dataset cora --num_exp 3
python train_teacher.py --model SAGE --dataset ogbn-arxiv --save_results --num_exp 3
python test_prompt.py --model SAGE --dataset cora --num_exp 3
```

### Result

| Dataset | End-to-end Learning | Prompt for Transfer Learning |
| :--- | :---: | :---: |
| CORA | 0.7827±0.0038 | 0.7770±0.0179 |
| CiteSeer | 0.6420±0.0036 | 0.6350±0.0119 |
| PubMed | 0.7890±0.0122 | 0.7610±0.0029 |

The table presents a comparison between two models: an end-to-end learning model using GraphSAGE and a Prompt for Transfer Learning model, also based on GraphSAGE. Both models were evaluated on three downstream datasets: CORA, CiteSeer, and PubMed. The end-to-end learning model is trained on the autoencoder preprocessed downstream dataset directly; the Prompt for Transfer Learning model is first pretrained on the ogbn-arxiv dataset, and then the prompt is trained on the autoencoder preprocessed downstream dataset. The results are presented as mean accuracy scores with standard deviations.

### Analysis

The end-to-end GraphSAGE model shows superior performance over the Prompt for Transfer Learning model in all three datasets, indicating that the traditional approach of utilizing preprocessed features directly for classification is more effective in these cases.

Since both models utilize the same preprocessed features, the comparison isolates the impact of the Prompt for Transfer Learning framework's additional mechanisms, such as the integration of the prompt $p$ and the use of class-prototypes for classification.

The consistent outperformance of the end-to-end model suggests that, in these scenarios, the prompt mechanism and the class-prototype based classification may not be adding beneficial information to the model. Instead, these mechanisms could be introducing complexity or noise that detracts from the model's ability to classify nodes accurately.

The evidence suggests that, at least for these datasets, the added complexity of the Prompt for Transfer Learning model does not translate to improved performance and may in fact hinder it. Further research could explore if there are specific types of datasets, or particular configurations of the Prompt for Transfer Learning framework, for which the model would outperform traditional end-to-end approaches.

## Conclusion

The Prompt for Transfer Learning framework presents a sophisticated approach to overcome feature and label space mismatches in node classification. By building on the commonalities between pretraining and downstream datasets and introducing a prompt-based mechanism, it paves the way for GNN models to generalize effectively across varying domains. The combination of graph autoencoders for feature alignment and similarity-based classification for label space unification encapsulates the adaptability of our framework, making it a potential solution for transferring knowledge in graph-based tasks.

However, the experimental results on the CORA, CiteSeer, and PubMed datasets, with both models trained on autoencoder preprocessed data, suggest that this approach may not always yield superior performance compared to traditional end-to-end models.

While the current analysis suggests limitations in the Prompt for Transfer Learning framework's ability to outperform a well-established end-to-end learning approach, it also opens avenues for refinement and exploration that could help realize the full potential of prompt-based transfer learning in graph neural networks.
