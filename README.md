# Question-Answering Project on Medical Papers

## Abstract
Question Answering (QA) is a field in the Natural Language Processing (NLP) and Information retrieval (IR). QA task basically aims to give precise and quick answers to given question in natural languages by using given data or databases. In this project, we tackled the problem of question answering on Medical Papers. There are plenty of Language Models published and available to use for Question Answering task. In this project, we wanted to develop a language model, specifically trained on Medical field. Our goal is to develop a context-specific language model on Medical papers, performs better than general language models. We used ELECTRA-small as our base model, and trained it using medical paper dataset, then fine-tuned on Medical QA dataset. 

You can access our med-electra small model here:
https://huggingface.co/enelpi/med-electra-small-discriminator


## Dataset
We used Medical Papers S2ORC. We filtered the S2ORC database using Field of Study, and took Medical papers. The dataset consists of shards, we took 13 shards of the Medical papers. After that, we took the ones which are published on PubMed and PubMEdCentral. We used only the pdf_parses of those papers, since sentences in the pdf_parses contains more information.

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
   
   Corpus Data Summary
|               |  Sentence  |      Vocabulary     |         Size        |
| ------------- |:----------:|:-------------------:|:-------------------:|
|     Train     | 111537350  |       27609654      |        16GB         |



## Model Training
Using the generated corpus, we pre-trained ELECTRA-small model from scratch. The model is trained on RTX 2080 Ti GPU. 

|      Model    |  Layers    |      Hidden Size    |     Parameters      |
| ------------- |:----------:|:-------------------:|:-------------------:|
| ELECTRA-Small |     12     |         256         |        14M          |


### ELECTRA-Small

| Model/Hyperparameters | epoch | max_seq_length | per_gpu_train_batch_size |
|:----------------------|:-----:|:--------------:|:------------------------:|
|     Electra-Small     |    -  |      -         |            -             |        

The training results can be accessed here:

https://tensorboard.dev/experiment/G9PkBFZaQeaCr7dGW2ULjQ/#scalars
https://tensorboard.dev/experiment/qu1bQ0MiRGOCgqbZHQs2tA/#scalars

You can see the loss graph here:
![Loss graph](/images/model_loss.png)


### RESULTS
|   Model/Score   |    F1    |    Exact   |  Loss Exact  |
|:----------------|:--------:|:----------:|:------------:|
| Electra         |     -    |     -      |      -       |   
| BERT, Cased     |     -    |     -      |      -       |
| BERT, Uncased   |     -    |       -    |      -       |

## Requirements
- Python
- Transformers
- Pytorch
- TensorFlow
- 

# References

https://github.com/google-research/electra
https://chriskhanhtran.github.io/_posts/2020-06-11-electra-spanish/
https://github.com/allenai/s2orc
https://github.com/allenai/scibert
https://github.com/abachaa/MedQuAD
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5530755/
https://github.com/LasseRegin/medical-question-answer-data
https://huggingface.co/blog/how-to-train
https://arxiv.org/abs/1909.06146
https://www.nlm.nih.gov/databases/download/pubmed_medline.html

