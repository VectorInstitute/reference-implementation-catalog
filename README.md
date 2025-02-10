# Reference Implementation Catalog

#TODO: Add an introduction here.

| Repository | Description | Algorithms  | Number of<br>datasets | Datasets  |
| :--------- | :---------- | :---------: | :--------------------:| :-------: |
| [laftr][lafter-repo] | <p>This repository contains code for the paper [Learning Adversarially Fair and Transferable Representations][laftr-paper] which was accepted at PMLR'18. <br> Authors are David Madras, Elliot Creager, Toniann Pitassi and Richard Zemel.</p> | laftr | 1 | [Adult] |
| [gram-ood-detection][god-repo] | This repository contains code for the paper [Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices][god-paper] which was accepted at ICML'20. <br> Authors are Chandramouli Shama Sastry, Sageev Oore. | OOD detection using Gram matrices | 7 | [CIFAR10], [CIFAR100], [SVHN]|
| [Computer_Vision_Project][cvp-repo] | This repository tackles different problems such as defect detection, footprint extraction, road obstacle detection, traffic incident detection, and segmentation of medical procedures. | Semantic segmentation using Unet, Unet++, FCN, DeepLabv3; Anomaly segmentation | 11 | [SpaceNet Building Detection V2], <br> [MVTEC], [ICDAR2015], [PASCAL_VOC] <br> [DOTA], [AVA], [UCF101-24] <br> [J-HMDB-21]|
| [Privacy Enhancing Technologies][pet-repo] | This repository contains demos for Privacy, Homomorphic Encryption, Horizontal and Vertical Federated Learning, MIA, and PATE | Vanilla SGD, DP SGD, DP Logistic Regression, Homomorphic Encryption for MLP, Horizontal FL, Horizontal FL on MLP, Membership Inference Attacks (MIA) using DP, MIA using SAM, PATE, Vertical FL. | 9 | [Heart Disease], [Credit Card Fraud] <br> [Breaset Cancer Data], [TCGA] <br> [CIFAR10][cifar10-pet], [Home Credit Default Risk] <br> [Yelp], [Airbnb]|
| [SSGVQAP][ssgvap-repo] | This repository contains code for the paper [A Smart System to Generate and Validate Question Answer Pairs for COVID-19 Literature][ssgvap-paper] which was accepted ibn ACL'20. Authors are Rohan Bhambhoria, Luna Feng, Dawn Sepehr, John Chen, Conner Cowling, Sedef Kocak, Elham Dolatabadi. | An Active Learning Strategy for Data Selection, AL-Uncertainty, AL-Clustering | 1 | [CORD-19] |
| [NeuralKernelBandits][nkb-repo] | This repository contains code for the paper [An Empirical Study of Neural Kernel Bandits][nkb-paper] which was accepted in Neurips'21. Authors are Lisicki, Michal, Arash Afkanpour, and Graham W. Taylor. | Neural tangent kernel, Conjugate kernel, NNGP, Deep ensembles, Randomized Priors, NTKGP, Upper Confidence Bounds (UCB), Thompson Sampling (TS) | 7 | [Mushroom], [Statlog] <br> [Adult][adult-nkb], [US Census 1990] <br> [Covertype] |
| [foodprice-forecasting][fpf-repo] | This repository replicates the experiments described on pages 16 and 17 of the [2022 Edition of Canada's Food Price Report][fpf-paper]. | Time series forecasting using Prophet,  Time series forecasting using Neural prophet, Interpretable time series forecasting using N-BEATS, Ensemble of the above methods. | 3 | [FRED Economic Data] |
| [Recommendation Systems][recsys-repo] | This repository contains demos for various RecSys techniques such as Collaborative Filtering, Knowledge Graph, RL based, Sequence Aware, Session based etc. | SVD++, NeuMF, Plot based, Two tower, SVD, KG based, SlateQ, BST, Simple Association Rules, first-order Markov Chains, Sequential Rules, RNN, Neural Attentive Session, BERT4rec, A2SVDModel, SLi-Rec | 7 | [Amazon-recsys] ,[careervillage], <br> [movielens-recsys], [tmdb], [LastFM] <br> [yoochoose] |
| [Forecasting with Deep Learning][forecasting-dl-repo] | This repository contains demos for a variety of forecasting techniques for Univariate and Multivariate time series, spatiotemporal forecasting etc. | Exponential Smoothing, Persistence Forecasting,  Mean Window Forecast,  Prophet, Neuralphophet, NBeats, DeepAR, Autoformer, DLinear, NHITS | 11 | [Canadian Weather Station Data], [BoC Exchange rate], [Electricity Consumption], [Road Traffic Occupancy], [Influenza-Like Illness Patient Ratios], [Walmart M5 Retail Product Sales], [WeatherBench], [Grocery Store Sales], [Economic Data with Food CPI] |
| [Prompt Engineering][pe-repo] |  This repository contains demos for a variety of Prompt Engineering techniques such as fairness measurement via sentiment analysis, finetuning, prompt tuning, prompt ensembling etc. | Bias Quantification & Probing, Stereotypical Bias Analysis, Binary sentiment analysis task, Finetuning using HF Library, Gradient-Search for Instruction Prefix, GRIPS for Instruction Prefix, LLM Summarization, LLM Classification: AG News task, LLM BoolQ (Boolean question answering), LLM Basic Translation (French -> English), LLM Aspect-Based Sentiment Analysis, prompt-tuning, Activation Computation, LLM Classifier Training, Voting and Averaging Ensembles | 10 | [Crow-pairs], [sst5], [cnn_dailymail], [ag_news], [Tweet-data], [Other] |
| [ABSA][absa-repo] | This repository contains code for the paper [Open Aspect Target Sentiment Classification with Natural Language Prompts][absa-paper]. <br> Authors are Ronald Seoh, Ian Birle, Mrinal Tak, Haw-Shiuan Chang, Brian Pinette, Alfred Hough. | Zero-shot inference for sentiment using PLM and openprompt, Few-shot inference for sentiment using PLM, Zero-shot ATSC with Prompts using BERT and OPT, Zero-shot inference of aspect term and generate sentiment polarity using NLTK pipeline | 4 | [Link][Link-absa] |
| [NAA][naa-repo] | This repository contains code for the paper [Bringing the State-of-the-Art to Customers: A Neural Agent Assistant Framework for Customer Service Support].[naa-paper] Authors are Stephen Obadinma, Faiza Khan Khattak, Shirley Wang, Tania Sidhorn, Elaine Lau, Sean Robertson, Jingcheng Niu, Winnie Au, Alif Munim, Karthik Raja Kalaiselvi Bhaskar. | Context Retreival using SBERT bi-encoder, Context Retreival using SBERT cross-encoder, Intent identification using BERT, Few Shot Multi-Class Text Classification with BERT, Multi-Class Text Classification with BERT, Reponse generation via GPT2. | 5 | [ELI5],  [MSMARCO] |
| [Anomaly Detection Project][anomaly-repo] | This repository contains demos for various supervised and unsupervised anomaly detection techniques in domains such as Fraud Detection, Network Intrusion Detection, System Monitoring and image, Video Analysis. | AMNet, GCN, SAGE, OCGNN, DON, AdONE, MLP, FTTransformter, DeepSAD, XGBoost, CBLOF, CFA for Target-Oriented Anomaly Localization, Draem for surface anomaly detection, Logistic Regression, CATBoost,  Random Forest, Diversity Measurable Anomaly Detection, Two-stream I3D Convolutional Network, DeepCNN, CatBoost, LighGBM, Isolation Forest, TabNet, AutoEncoder, Internal Contrastive Learning | 5 | [On Vector Cluster][cluster-anomaly] |
| [SSL Bootcamp][ssl-repo] | This repository contains demos for self-supervised techniques such as contrastive learning, masked modeling and self distillation. | Internal Contrastive Learning, LatentOD-AD, TabRet,SimMTM, Data2Vec | 52 | [Beijing Air Quality][baq-ssl], [BRFSS][brfss-ssl], [Stroke Prediction][stroke-ssl], [STL10][stl-10-ssl], [Link1][Link1-ssl], [Link2][Link2-ssl]
| [Causal Inference Lab][ci-lab-repo] |  This repository contains code to estimate the causal effects of an intervention on some measurable outcome primarily in the health domain. | Naive ATE, TARNet, DragonNet, Double Machine Learning, T Learner, S Learner, Inverse Propensity based Learner, PEHE, MAE; Evaluation metrics: R Score, DR Score, Tau Risk, Tau IPTW Score, Tau DR Score, Tau S Score, Tau T Risk, Influence Score | 5 | [Infant Health and Development Program][IHDP], <br> [Jobs], [Twins], <br> [Berkeley admission], <br> [Government Census], [Compas] |
| [VariationalNeuralAnnealing][vna-repo] | This repository contains code for the paper [Variational neural annealing][vna-paper]. Authors are Mohamed Hibat-Allah, Estelle M. Inack, Roeland Wiersema, Roger G. Melko & Juan Carrasquilla. | Variational neural annealing; Variational Classical Annealing (VCA), Variational Quantum Annealing, Regularized VQA, Classical-Quantum Optimization | 2 | [Edwards-Anderson][EA], [Sherrington-Kirkpatrick][SK] |
| [HV-Ai-C][hvaic-repo] | This repository implements a Reinforcement Learning agent to optimize energy consumption within Data Centers. | RL agents performing Random action, Fixed action, Q Learning; Hyperspace Neighbor Penetration | - | - |
| [Flex Model][flex-model-repo] | This repository contains code for the paper [FlexModel: A Framework for Interpretability of Distributed Large Language Models][flex-model-paper]. Authors are Matthew Choi, Muhammad Adil Asif, John Willes, David Emerson.| Distributed Interpretability | - | - |
| [VBLL][vbll-repo] | This repository contains code for the paper [Variational Bayesian Last Layers][vbll-paper]. Authors are James Harrison, John Willes, Jasper Snoek. | Variational Bayesian Last Layers | 2 | [MNIST], [FashionMNIST] |
| [RAG][rag-repo] | This repository contains demos for various Retrieval Augmented Generation techniques using different libraries. | Cloud search via LlamaHub,  Document search via LangChain, LlamaIndex for OpenAI and Cohere models, Hybrid Search via Weaviate Vector Store, Evaluation via RAGAS library, Websearch via LangChain | 3 | [Vectors 2021 Annual Report], [PubMed Doc], [Banking Deposits] |
| [Finetuning and Alignment][fa-repo] | This repository contains demos for finetuning techniques for LLMs focussed on reducing computational cost. | DDP, FSDP, Instruction Tuning, LoRA, DoRA, QLora,Supervised finetuning | 3 | [samsam], [imdb], [Bias-DeBiased] |
| [Prompt Engineering Laboratory][pe-lab-repo] | This repository contains demos for various Prompt Engineering techniques, along with examples for Bias quantification, text classification. | Stereotypical Bias Analysis, Sentiment inference, Finetuning using HF Library, Activation Generation, Train and Test Model for Activations without Prompts, RAG, ABSA, Few shot prompting, Zero shot prompting (Stochastic, Greedy, Likelihood Estimation), Role play prompting, LLM Prompt Summarization, Zero shot and few shot prompt translation, Few shot CoT, Zero shot CoT, Self-Consistent CoT prompting (Zero shot, 5-shot), Balanced Choice of Plausible Alternatives, Bootstrap Ensembling(Generation & MC formulation), Vote Ensembling. | 11 | [Crows-pairs][crow-pairs-pe-lab], [sst5][sst5-pe-lab], [czarnowska templates][czar-templ-pe-lab], [cnn_dailymail], [ag_news], [Weather and sports data], [Other] |
| [bias-mitigation-unlearning][bmu-repo] | This repository contains code for the paper [Can Machine Unlearning Reduce Social Bias in Language Models?][bmu-repo] which was published at EMNLP'24 in the Industry track. Authors are Omkar Dige, Diljot Arneja, Tsz Fung Yau, Qixuan Zhang, Mohammad Bolandraftar, Xiaodan Zhu, Faiza Khan Khattak. | PCGU, Task vectors and DPO for Machine Unlearning | 20 | [BBQ][bbq-bmu], [Stereoset][stereoset-bmu], [Link1][link1-bmu], [Link2][link2-bmu] |
[cyclops-workshop][cyclops-repo] | This repository contains demos for using [CyclOps] package for clinical ML evaluation and monitoring. | Xgboost | 1 | [Diabetes 130-US hospitals dataset for years 1999-2008][diabetes-cyclops] |
[odessey][odessey-repo] | This is a library created with research done for the paper [EhrMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records][odessey-paper] published at ArXiv'24. |EhrMamba, Xgboost, Bi-LSTM | 1 | [MIMIC-IV] |
[Diffusion model bootcamp][diffusion-repo] | This repository contains demos for various diffusion models for tabular and time series data. | TabDDPM, TabSyn, ClavaDDPM, CSDI, TSDiff | 12 | [Physionet Challenge 2012], [wiki2000] | 
[News Media Bias][nmb-repo] | This repository contains code for libraries and experiments to recognise and evaluate bias and fakeness within news media articles via LLMs. | Bias evaluation via LLMs, finetuning and data annotation via LLM for fake news detection, Supervised finetuning for debiasing sentence, NER for biased phrases via LLMS, Evaluate using DeepEval library. | 4 | [News Media Bias Full data][nmb-data], [Toxigen], [Nela GT], [Debiaser data] |
--------

**DISCLAIMER**
- Many repositories contain code for reference purposes only. In order to run them, updates may be required to the code and environment files. 
- Links for only publicly available datasets are provided. Many datasets used in the repositories are only available on the the Vector cluster.


[//]: # (Reference links for Github repositories)
[lafter-repo]: https://github.com/VectorInstitute/laftr
[god-repo]: https://github.com/VectorInstitute/gram-ood-detection
[cvp-repo]: https://github.com/VectorInstitute/Computer_Vision_Project
[pet-repo]: https://github.com/VectorInstitute/PETs-Bootcamp
[ssgvap-repo]: https://github.com/VectorInstitute/SSGVQAP
[nkb-repo]: https://github.com/VectorInstitute/NeuralKernelBandits
[fpf-repo]: https://github.com/VectorInstitute/foodprice-forecasting
[recsys-repo]: https://github.com/VectorInstitute/recommender_systems_project
[forecasting-dl-repo]: https://github.com/VectorInstitute/forecasting-with-dl
[pe-repo]: https://github.com/VectorInstitute/PromptEngineering
[fastgan-repo]: https://github.com/VectorInstitute/FastGAN-pytorch
[absa-repo]: https://github.com/VectorInstitute/ABSA
[naa-repo]: https://github.com/VectorInstitute/NAA
[anomaly-repo]: https://github.com/VectorInstitute/anomaly-detection-project
[ssl-repo]: https://github.com/VectorInstitute/SSL-Bootcamp
[ci-lab-repo]: https://github.com/VectorInstitute/Causal_Inference_Laboratory
[vna-repo]: https://github.com/VectorInstitute/VariationalNeuralAnnealing
[covid-repo]: https://github.com/VectorInstitute/ProjectLongCovid-NER
[hvaic-repo]: https://github.com/VectorInstitute/HV-Ai-C
[flex-model-repo]: https://github.com/VectorInstitute/flex_model
[vbll-repo]: https://github.com/VectorInstitute/vbll
[rag-repo]: https://github.com/VectorInstitute/rag_bootcamp
[fa-repo]: https://github.com/VectorInstitute/finetuning-and-alignment
[pe-lab-repo]: https://github.com/VectorInstitute/PromptEngineeringLaboratory
[bmu-repo]: https://github.com/VectorInstitute/bias-mitigation-unlearning
[cyclops-repo]: https://github.com/VectorInstitute/cyclops-workshop
[odessey-repo]: https://github.com/VectorInstitute/odyssey
[nmb-repo]: https://github.com/VectorInstitute/news-media-bias
[diffusion-repo]: https://github.com/VectorInstitute/diffusion_model_bootcamp

[//]: # (Reference links for Research papers)
[laftr-paper]: https://arxiv.org/abs/1802.06309
[god-paper]: http://proceedings.mlr.press/v119/sastry20a.html
[ssgvap-paper]: https://aclanthology.org/2020.sdp-1.4/
[nkb-paper]: https://arxiv.org/abs/2111.03543
[fpf-paper]: https://www.dal.ca/sites/agri-food/research/canada-s-food-price-report-2022.html
[absa-paper]: https://aclanthology.org/2021.emnlp-main.509/
[vna-paper]: https://www.nature.com/articles/s42256-021-00401-3
[flex-model-paper]: https://arxiv.org/abs/2312.03140
[vbll-paper]: https://arxiv.org/abs/2404.11599
[bmu-paper]: https://aclanthology.org/2024.emnlp-industry.71/
[odessey-paper]: https://arxiv.org/pdf/2405.14567

[//]: # (Reference links for datasets)
[CIFAR10]: https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10
[CIFAR100]: https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100
[SVHN]: https://pytorch.org/vision/main/generated/torchvision.datasets.SVHN.html#torchvision.datasets.SVHN
[Adult]: https://github.com/VectorInstitute/laftr/tree/master/data/adult
[SpaceNet Building Detection V2]: https://spacenet.ai/spacenet-buildings-dataset-v2/
[MVTEC]: https://www.mvtec.com/company/research/datasets/mvtec-ad
[ICDAR2015]: https://drive.google.com/drive/folders/12eg7u7oBkZ6-ov3ITiED4nLlQzP4KoTd
[PASCAL_VOC]: https://drive.google.com/drive/folders/12eg7u7oBkZ6-ov3ITiED4nLlQzP4KoTd
[DOTA]: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly
[AVA]: https://github.com/cvdfoundation/ava-dataset
[UCF101-24]: https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view?usp=sharing
[J-HMDB-21]: http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets
[Heart Disease]: https://www.kaggle.com/datasets/ronitf/heart-disease-uci
[Credit Card Fraud]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[Breaset Cancer Data]: https://github.com/VectorInstitute/PETs-Bootcamp/blob/main/HE_TenSEAL/breast_cancer_data.csv
[TCGA]: https://vectorinstituteai-my.sharepoint.com/personal/sedef_kocak_vectorinstituteai_onmicrosoft_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsedef%5Fkocak%5Fvectorinstituteai%5Fonmicrosoft%5Fcom%2FDocuments%2FPETS%5FProject%5FParticipants%2FExample%20Datasets%2FKidney%20Histopathology&ga=1
[cifar10-pet]: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10
[Home Credit Default Risk]: https://www.kaggle.com/c/home-credit-default-risk/overview
[Yelp]: https://business.yelp.com/data/resources/open-dataset/
[Airbnb]: https://insideairbnb.com/get-the-data/
[CORD-19]: https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge
[Mushroom]: https://archive.ics.uci.edu/dataset/73/mushroom
[Statlog]: https://github.com/VectorInstitute/NeuralKernelBandits/tree/main/contextual_bandits/datasets
[Adult-nkb]: https://archive.ics.uci.edu/dataset/2/adult
[US Census 1990]: https://archive.ics.uci.edu/dataset/116/us+census+data+1990
[Covertype]: https://archive.ics.uci.edu/dataset/31/covertype
[FRED Economic Data]: https://fred.stlouisfed.org/
[Amazon-recsys]: https://drive.google.com/drive/folders/1w9ofYRBZN5XIb8M-UzbU3Wp4H1ZBYOFi?usp=drive_link
[careervillage]: https://drive.google.com/drive/folders/1rNeBtNYM7Z0oHVho75PDEP3VZIXAoxx9?usp=drive_link
[movielens-recsys]: https://drive.google.com/drive/folders/112OtYq83WZgVqV43pGhKZjVTzlUhKM-b?usp=drive_link
[tmdb]: https://drive.google.com/drive/folders/1CU863OynVNnNTTduKxExubCyJLCZPF0R?usp=drive_link
[LastFM]: https://drive.google.com/drive/folders/1Jftz1_olxblJVZe6ZDMdrclAW_YnCOci?usp=drive_link
[yoochoose]: https://drive.google.com/drive/folders/1XNyPH8i-pxnNbJKjZZRCL1oz-HPZscLC?usp=drive_link
[Canadian Weather Station Data]: https://drive.google.com/drive/folders/1YeOoJNf7VCy7r3sFhdTrl7WdevcUIZNW
[BoC Exchange rate]: https://drive.google.com/drive/folders/1Z9pnC0kPN-c_eAHSsPyWPYRnnGR3sEuf
[Electricity Consumption]: https://drive.google.com/drive/folders/1YIl6RHAQ5muZEjFjXLj7Zt4vOwKUu2Qe
[Road Traffic Occupancy]: https://drive.google.com/drive/folders/1YDM-mMGuhlE_pTlwb5qoPcOQfspJ4m4W
[Influenza-Like Illness Patient Ratios]: https://drive.google.com/drive/folders/1YFoC3fWY-22S11MtfKHnl_R8OminZ2eo
[Walmart M5 Retail Product Sales]: https://drive.google.com/drive/folders/1bc488T1GsJ3xg2nQmuFTut7uF1SFDSp2
[WeatherBench]:https://drive.google.com/drive/folders/1YD-Hadx_T4JZcjmvFYDp4Pb71852CIVT
[Grocery Store Sales]: https://drive.google.com/drive/folders/1as_cJgJbzw1OlnWyF8Y3xj7ZEjRh_kD6
[Economic Data with Food CPI]: https://drive.google.com/drive/folders/1cNyHR5DpUQ5RORgDS8pB8iswWo-iBLFI
[Crow-pairs]: https://github.com/VectorInstitute/PromptEngineering/blob/main/src/reference_implementations/fairness_measurement/crow_s_pairs/resources/crows_pairs_anonymized.csv
[sst5]: http://github.com/VectorInstitute/PromptEngineering/blob/main/src/reference_implementations/fairness_measurement/czarnowska_analysis/resources/processed_sst5.tsv
[cnn_dailymail]: https://huggingface.co/datasets/ccdv/cnn_dailymail
[ag_news]: https://huggingface.co/datasets/fancyzhx/ag_news
[Tweet-data]: https://github.com/VectorInstitute/PromptEngineering/tree/main/resources/datasets
[Other]: https://github.com/VectorInstitute/PromptEngineering/tree/main/src/reference_implementations/prompting_vector_llms/llm_prompting_examples/resources
[Few shot images dataset]: https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view
[Link-absa]: https://github.com/VectorInstitute/ABSA/tree/main/atsc_paper/atsc_prompts_modified/dataset_files
[ELI5]: https://drive.google.com/drive/folders/1PDBiij-6JSxOtplOSc0hPTk9zL9n3qR6
[MSMARCO]: https://drive.google.com/drive/folders/1LO3OtuDC_FSFktTgb2NfjPY2cse7WcTY
[cluster-anomaly]: https://github.com/VectorInstitute/anomaly-detection-project/tree/main?tab=readme-ov-file#datasets
[baq-ssl]: https://zenodo.org/records/3902671
[brfss-ssl]: https://www.cdc.gov/brfss/
[stroke-ssl]: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
[stl-10-ssl]: https://cs.stanford.edu/~acoates/stl10/
[Link1-ssl]: https://github.com/VectorInstitute/SSL-Bootcamp/tree/main/contrastive_learning/ICL/datasets/Classical
[Link2-ssl]: https://github.com/VectorInstitute/SSL-Bootcamp/tree/main/contrastive_learning/LatentOE/DATA
[IHDP]: https://github.com/VectorInstitute/Causal_Inference_Laboratory/tree/main/data/IHDP-100
[Jobs]: https://github.com/VectorInstitute/Causal_Inference_Laboratory/tree/main/data/Jobs
[Twins]: https://github.com/VectorInstitute/Causal_Inference_Laboratory/tree/main/data/TWINS
[Berkeley admission]: https://github.com/VectorInstitute/Causal_Inference_Laboratory/tree/main/data/CFA
[Government Census]: https://github.com/VectorInstitute/Causal_Inference_Laboratory/tree/main/data/CFA 
[Compas]: https://github.com/VectorInstitute/Causal_Inference_Laboratory/tree/main/data/CFA
[EA]: https://github.com/VectorInstitute/VariationalNeuralAnnealing/tree/main/data/EA
[SK]: https://github.com/VectorInstitute/VariationalNeuralAnnealing/tree/main/data/SK
[MNIST]: https://huggingface.co/datasets/ylecun/mnist
[FashionMNIST]: https://huggingface.co/datasets/zalando-datasets/fashion_mnist
[Vectors 2021 Annual Report]: https://github.com/VectorInstitute/rag_bootcamp/tree/main/cloud_search/source_documents
[PubMed Doc]: https://github.com/VectorInstitute/rag_bootcamp/tree/main/pubmed_qa/data/pubmed_doc
[Banking Deposits]: https://github.com/VectorInstitute/rag_bootcamp/tree/main/sql_search
[samsam]: https://huggingface.co/datasets/Samsung/samsum
[imdb]: https://github.com/VectorInstitute/finetuning-and-alignment/tree/main/data/imdb
[Bias-DeBiased]: https://github.com/VectorInstitute/finetuning-and-alignment/tree/main/data
[crow-pairs-pe-lab]: https://github.com/VectorInstitute/PromptEngineeringLaboratory/tree/main/src/reference_implementations/fairness_measurement/crow_s_pairs/resources
[sst5-pe-lab]: https://github.com/VectorInstitute/PromptEngineeringLaboratory/tree/main/src/reference_implementations/fairness_measurement/czarnowska_analysis/resources
[czar-templ-pe-lab]: https://github.com/VectorInstitute/PromptEngineeringLaboratory/tree/main/src/reference_implementations/fairness_measurement/resources/czarnowska_templates
[cnn_dailymail]: https://huggingface.co/datasets/ccdv/cnn_dailymail
[ag_news]: https://huggingface.co/datasets/fancyzhx/ag_news
[Weather and sports data]: https://github.com/VectorInstitute/PromptEngineeringLaboratory/tree/main/src/reference_implementations/prompting_vector_llms/llm_basic_rag_example/resources
[link1-bmu]: https://github.com/VectorInstitute/bias-mitigation-unlearning/tree/main/reddit_bias/data
[bbq-bmu]: https://github.com/nyu-mll/BBQ/tree/main/data
[stereoset-bmu]: https://github.com/moinnadeem/StereoSet/tree/master/data
[link2-bmu]: https://github.com/VectorInstitute/bias-mitigation-unlearning/tree/main/task_vectors/src/datasets
[diabetes-cyclops]: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
[MIMIC-IV]: https://physionet.org/content/mimiciv/2.0/
[nmb-data]: https://huggingface.co/datasets/newsmediabias/news-bias-full-data
[Toxigen]: https://github.com/VectorInstitute/news-media-bias/blob/main/Evaluations/toxigen_eval/
[Nela GT]: https://dataverse.harvard.edu/file.xhtml?fileId=6078140&version=2.0
[Debiaser data]: https://github.com/VectorInstitute/news-media-bias/tree/main/UnBIAS-Debiaser%20library/datasets
[Physionet Challenge 2012]: https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download
[wiki2000]: https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz

[//]: # (Miscellaneous reference links)

[CyclOps]: https://github.com/VectorInstitute/cyclops