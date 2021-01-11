# Medical Field Specific Electra Model

## Abstract
Question Answering (QA) is a field in the Natural Language Processing (NLP) and Information retrieval (IR). QA task basically aims to give precise and quick answers to given question in natural languages by using given data or databases. In this project, we tackled the problem of question answering on Medical Papers. There are plenty of Language Models published and available to use for Question Answering task. In this project, we wanted to develop a language model, specifically trained on Medical field. Our goal is to develop a context-specific language model on Medical papers, performs better than general language models. We used ELECTRA-small as our base model, and trained it using medical paper dataset, then fine-tuned on Medical QA dataset. We trained three different models, and compared their results on the NLP downstream tasks.

You can access our models here:



med-electra small model 17GB - 64k vocab
https://huggingface.co/enelpi/med-electra-small-discriminator

med-electra small model 39GB - 30.5k vocab


med-electra small model 39GB - 64k vocab


## Datasets
We used Medical Papers S2ORC. We filtered the S2ORC database using Field of Study, and took Medical papers. We used two different datasets,  consists of shards, we took 11 shards for the 17GB dataset, and used 26 shards for 39GB dataset. After that, we took the ones which are published on PubMed and PubMEdCentral. We used only the pdf_parses of those papers, since sentences in the pdf_parses contains more information.

```json{'paper_id': '1',
{
    "section": "Introduction",
    "text": "Dogs are happier cats [13, 15]. See Figure 3 for a diagram.",
    "cite_spans": [
        {"start": 22, "end": 25, "text": "[13", "ref_id": "BIBREF11"},
        {"start": 27, "end": 30, "text": "15]", "ref_id": "BIBREF30"},
        ...
    ],
    "ref_spans": [
        {"start": 36, "end": 44, "text": "Figure 3", "ref_id": "FIGREF2"},
    ]
}
{
    ...,
    "BIBREF11": {
        "title": "Do dogs dream of electric humans?",
        "authors": [
            {"first": "Lucy", "middle": ["Lu"], "last": "Wang", "suffix": ""}, 
            {"first": "Mark", "middle": [], "last": "Neumann", "suffix": "V"}
        ],
        "year": "", 
        "venue": "barXiv",
        "link": null
    },
    ...
}
{
    "TABREF4": {
        "text": "Table 5. Clearly, we achieve SOTA here or something.",
        "type": "table"
    }
    ...,
    "FIGREF2": {
        "text": "Figure 3. This is the caption of a pretty figure.",
        "type": "figure"
    },
    ...
}
} 
   ```
   
   Corpus Data Summary for 17GB
|               |  Sentence  |      Unique Words     |         Size        |    Token Size   |
| ------------- |:----------:|:---------------------:|:-------------------:|:----------------|
|     Train     | 111537350  |       27609654        |        16.9GB       |    2538210492   |


   Corpus Data Summary for 39GB
|               |  Sentence  |      Unique Words     |         Size        |    Token Size   |
| ------------- |:----------:|:---------------------:|:-------------------:|:----------------|
|     Train     | 263134203  |       52206886        |        39.9GB       |    6000436472   |


## Model Training
Using the generated corpus, we pre-trained ELECTRA-small model from scratch. The model is trained on RTX 2080 Ti GPU. 

|      Model    |  Layers    |      Hidden Size    |     Parameters      |
| ------------- |:----------:|:-------------------:|:-------------------:|
| ELECTRA-Small |     12     |         256         |        14M          |


For 17GB

Number of Lines: 111332331

Number of words(tokens): 2538210492

|      Metric                  |     Value    |
| -----------------------------|:------------:|
| disc_accuracy                |     0.9456   |
| disc_auc                     |     0.9256   |
| disc_loss                    |     0.154    |
| disc_precision               |     0.7832   |
| disc_recall                  |     0.4545   |
| loss                         |     10.45    |
| maked_lm_accuracy            |     0.5168   |
| maked_lm_loss                |     2.776    |
| sampled_masked_lm_accuracy   |     0.4135   |




### ELECTRA-Small

For 17GB 

| Model/Hyperparameters | train_steps | vocab_size | batch_size |
|:----------------------|:-----------:|:----------:|:----------:|
|     Electra-Small     |    1M       |     64000  |   128      |        

The training results can be accessed here:

https://tensorboard.dev/experiment/G9PkBFZaQeaCr7dGW2ULjQ/#scalars
https://tensorboard.dev/experiment/qu1bQ0MiRGOCgqbZHQs2tA/#scalars


![Loss graph](/images/model_loss.png)


### RESULTS
### Named Entity Recognition
For named entity recognition,we used NCBI-Disease corpus, which is a resource for disease name recognition published in PubMed. 

#### Sequence Length 128

|   Model                                |    F1    |    Loss    |  accuracy    |  precision  |  recall  |
|:---------------------------------------|:--------:|:----------:|:------------:|:-----------:|:--------:|
| enelpi/med-electra-small-discriminator |  0.8462  |   0.0545   |     0.9827   |    0.8052   |  0.8462  |
| google/electra-small-discriminator     |  0.8294  |   0.0640   |     0.9806   |    0.7998   |  0.8614  |
| google/electra-base-discriminator      |  0.8580  |   0.0675   |     0.9835   |    0.8446   |  0.8718  |
| distilbert-base-uncased                |  0.8348  |   0.0832   |     0.9815   |    0.8126   |  0.8583  |
| distilroberta-base                     |  0.8416  |   0.0828   |     0.9808   |    0.8207   |  0.8635  |


#### Sequence Length 192

|   Model                                |    F1    |    Loss    |  accuracy    |  precision  |  recall  |
|:---------------------------------------|:--------:|:----------:|:------------:|:-----------:|:--------:|
| enelpi/med-electra-small-discriminator |  0.8425  |   0.0545   |     0.9824   |    0.8028   |  0.8864  |
| google/electra-small-discriminator     |  0.8280  |   0.0642   |     0.9807   |    0.7961   |  0.8625  |
| google/electra-base-discriminator      |  0.8648  |   0.0682   |     0.9838   |    0.8442   |  0.8864  |
| distilbert-base-uncased                |  0.8373  |   0.0806   |     0.9814   |    0.8153   |  0.8604  |
| distilroberta-base                     |  0.8329  |   0.0811   |     0.9801   |    0.8100   |  0.8572  |


#### Sequence Length 256

|   Model                                |    F1    |    Loss    |  accuracy    |  precision  |  recall  |
|:---------------------------------------|:--------:|:----------:|:------------:|:-----------:|:--------:|
| enelpi/med-electra-small-discriminator |  0.8463  |   0.0559   |     0.9823   |    0.8071   |  0.8895  |
| google/electra-small-discriminator     |  0.8280  |   0.0691   |     0.9806   |    0.8025   |  0.8552  |
| google/electra-base-discriminator      |  0.8542  |   0.0645   |     0.9840   |    0.8307   |  0.8791  |
| distilbert-base-uncased                |  0.8424  |   0.0799   |     0.9822   |    0.8251   |  0.8604  |
| distilroberta-base                     |  0.8339  |   0.0924   |     0.9806   |    0.8136   |  0.8552  |


### Question Answering
For Question Answering task, we used BioASQ question dataset.

|   Model                                    |    SAcc  |    LAcc    |
|:-------------------------------------------|:--------:|:----------:|
| enelpi/med-electra-small-discriminator-128 |  0.2821  |   0.4359   |
| google/electra-small-discriminator-128     |  0.3077  |   0.5128   |
| enelpi/med-electra-small-discriminator-512 |  0.1538  |   0.3590   |
| google/electra-small-discriminator-512     |  0.2564  |   0.5128   |

## Presentation
You can access presentation video from YouTube.
https://www.youtube.com/watch?v=faO9cLYFLdc&list=PLhnxo6HZwBglge_IYwGyXNmpz-3pGPggt&index=2


## Requirements
- Python
- Transformers
- Pytorch
- TensorFlow

# References

https://github.com/google-research/electra
https://chriskhanhtran.github.io/_posts/2020-06-11-electra-spanish/
https://github.com/allenai/s2orc
https://github.com/allenai/scibert
https://github.com/abachaa/MedQuAD
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5530755/
https://pubmed.ncbi.nlm.nih.gov/24393765/
https://github.com/LasseRegin/medical-question-answer-data
https://huggingface.co/blog/how-to-train
https://arxiv.org/abs/1909.06146
https://www.nlm.nih.gov/databases/download/pubmed_medline.html

