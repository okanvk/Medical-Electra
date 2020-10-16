# Question-Answering Project on Medical Papers

## Abstract
Question Answering (QA) is a field in the Natural Language Processing (NLP) and Information retrieval (IR). QA task basically aims to give precise and quick answers to given question in natural languages by using given data or databases. In this project, we tackled the problem of question answering on Medical Papers. There are plenty of Language Models published and available to use for Question Answering task. In this project, we wanted to develop a language model, specifically trained on Medical field. Our goal is to develop a context-specific language model on Medical papers, performs better than general language models. We used ELECTRA-small as our base model, and trained it using medical paper dataset, then fine-tuned on Medical QA dataset. 


## Dataset
We used Medical Papers S2ORC. We filtered the S2ORC database using Field of Study, and took Medical papers. The dataset consists of shards, we took 13 shards of the Medical papers. After that, we took the ones which are published on PubMed and PubMEdCentral. We used only the pdf_parses of those papers, since sentences in the pdf_parses contains more information.


```json{'paper_id': '1',
 '_pdf_hash': 'example text',
 'abstract': [],
 'body_text': [{'section': '',
   'text': 'example text',
   'cite_spans': [],
   'ref_spans': [{'start': 1,
     'end': 1,
     'text': 'Table I',
     'ref_id': 'TABREF1'}]},
  {'section': 'example text',
   'text': "example text",
   'cite_spans': [],
   'ref_spans': []},
  {'section': 'example text',
   'text': 'example text',
   'cite_spans': [{'start': 1,
     'end': 1,
     'text': 'example text',
     'ref_id': 'BIBREF11'}],
   'ref_spans': []},
  {'section': 'example text',
   'text': 'example text',
   'cite_spans': [],
   'ref_spans': []},
  {'section': 'example text',
   'text': 'example text',
   'cite_spans': [],
   'ref_spans': []},
  {'section': 'example text',
   'text': 'example text',
   'cite_spans': [],
   'ref_spans': []}],
 'bib_entries': {'BIBREF1': {'title': 'example text',
   'authors': [],
   'year': '',
   'venue': '',
   'link': None}
   'year': 1,
   'venue': 'example text',
   'link': '1'}},
 'ref_entries': {'FIGREF0': {'text': 'example text',
   'type': 'example text'},
  'TABREF0': {'text': 'example text',
   'type': 'table'},
  'TABREF1': {'text': 'example text',
   'type': 'table'},
  'TABREF2': {'text': 'example text',
   'type': 'table'},
  'TABREF3': {'text': 'example text',
   'type': 'table'},
  'TABREF4': {'text': 'example text',
   'type': 'table'},
  'TABREF5': {'text': 'example text',
   'type': 'table'}}} 
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

## Requirements
- Python
- Transformers
- Pytorch
- TensorFlow
- 

# References

ELECTRA
https://github.com/google-research/electra
https://chriskhanhtran.github.io/_posts/2020-06-11-electra-spanish/



S2ORC
https://github.com/allenai/s2orc

SciBERT
https://github.com/allenai/scibert

COQA 
paper: https://www.aclweb.org/anthology/Q19-1016.pdf
website: https://stanfordnlp.github.io/coqa/

Large QA datasets: https://github.com/ad-freiburg/large-qa-datasets

ORQA
https://arxiv.org/pdf/1906.00300.pdf
https://github.com/google-research/language/tree/master/language/orqa
https://github.com/google-research/language/tree/master/language/realm

QuAC
https://quac.ai/

MedQuAD dataset
https://github.com/abachaa/MedQuAD

arXiv dataset
https://www.kaggle.com/Cornell-University/arxiv

emrQA
https://www.aclweb.org/anthology/D18-1258/

Medical question answering for clinical decision support
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5530755/

MEANS
https://www.sciencedirect.com/science/article/abs/pii/S0306457315000515

HotpotQA
https://hotpotqa.github.io/

QAngaroo
http://qangaroo.cs.ucl.ac.uk/index.html

SciQ Dataset
https://allenai.org/data/sciq

Medical-question-answer-data (Çeşitli sitelerden toplanmış medical question answer dataseti)
https://github.com/LasseRegin/medical-question-answer-data

Hugging Face Train a Language Model From Scratch
https://huggingface.co/blog/how-to-train

PubMedQA (Biomedical QA for yes/no type questions)
https://arxiv.org/abs/1909.06146

PubMed Dataset (PubMed de yayınlanan makaleler citation bilgileri ve abstractları ile)
https://www.nlm.nih.gov/databases/download/pubmed_medline.html

Talk to Papers: Bringing Neural Question Answering to Academic Search. https://arxiv.org/abs/2004.02002 . Sanırım bizim yapmak istediğimize en yakın bu. SOCO bir framework .API ve dokümanı varmış. (https://docs.soco.ai/ )” The answer and question encoder are trained on publicly available QA datasets, including SQuAD (Rajpurkar et al., 2016), Natural Questions (Kwiatkowski et al., 2019) and MSMARCO (Nguyen et al., 2016)” → Bu veri setlerini kullanmışlar.

Can Machines Learn to Comprehend Scientific Literature? (https://ieeexplore.ieee.org/document/8606080) → PaperQA veri seti kullanılmış. Linkler çalışmıyor. Google’da aratınca Kaggle’da bir tane veri seti çıkıyor (https://www.kaggle.com/c/ee448-paperqa/ ) ama sadece davet edilen kulanıcılar katılabiliyor ve veri setlerini görebiliyor.” This is a limited-participation competition. Only invited users may participate.”

Veri madenciliği ödevi olarak Github’da BERT ile kullanımını buldum. Ama veri setini orada da göremedim. https://github.com/ImCharlesY/Data-Mining-Assignments/tree/cc6b13c368421a8d988625c71b6f47ab7c1785e2/paper-qa 
Allenai’de veri seti aramasına academic yazınca çıkan veri setleri.

SciDocs
Academic paper representation dataset accompanying the SPECTER paper/model
S2ORC: The Semantic Scholar Open Research Corpus
The largest collection of machine-readable academic papers to date for NLP & text mining.


Generating Scientific Question Answering Corpora from Q&A forums
https://paperswithcode.com/paper/generating-scientific-question-answering/review/ 
Kodları ve veri setini yine göremedim.SciQA. “We demonstrated our approach on three forums that are relevant to biomedical sciences: Biology from Stack Exchange, Medical Sciences also from Stack Exchange, and Nutrition from Reddit.” SciQA corpusu, üç farklı Soru-Cevap forumundan elde edilen 5.433 soru ve 10.204 soru-makale çiftinden oluşmaktadır.

Medikal question-answering için → https://github.com/abachaa/MedQuAD 


Deep learning.ai’ın Pie & AI: Real-world AI Applications in Medicine (https://youtu.be/Rp7qqjlBeRY?t=5482 ) ‘de bahseden kişi. Kişiyi Googleyınca
https://www.coursera.org/lecture/ai-for-medical-treatment/medical-question-answering-rDNFu - Coursera Medical question answering
