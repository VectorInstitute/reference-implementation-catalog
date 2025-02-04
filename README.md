# Reference Implementation Catalog

#TODO: Add an introduction here.

| Repository | Description | Algorithms  | Number of<br>datasets | Datasets  |
| :--------- | :---------- | :---------: | :--------------------:| :-------: |
| [laftr][lafter-repo] | This repository contains code for the paper [Learning Adversarially Fair and Transferable Representations][laftr-paper] , which was accepted at PMLR'18. <br> Authors are David Madras, Elliot Creager, Toniann Pitassi and Richard Zemel. | laftr | 1 | [Adult] |
| [gram-ood-detection][god-repo] | This repository contains code for the paper [Detecting Out-of-Distribution Examples with In-distribution Examples and Gram Matrices][god-paper] which was accepted at ICML'20. <br> Authors are Chandramouli Shama Sastry, Sageev Oore. | OOD detection using Gram matrices | 7 | [CIFAR10] [CIFAR100] [SVHN]|
| [Computer_Vision_Project][cvp-repo] | This repository tackles different problems such as defect detection, footprint extraction, road obstacle detection, traffic incident detection, and segmentation of medical procedures. | Semantic segmentation using Unet, Unet++, FCN, DeepLabv3; Anomaly segmentation | 11 | [SpaceNet Building Detection V2] [MVTEC] [ICDAR2015] [PASCAL_VOC] [DOTA] [AVA] [UCF101-24] [J-HMDB-21]|
| [Privacy Enhancing Technologies][pet-repo] | This repository contains demos for Privacy, Homomorphic Encryption, Horizontal and Vertical Federated Learning, MIA, and PATE | Vanilla SGD, DP SGD, DP Logistic Regression, Homomorphic Encryption for MLP, Horizontal FL, Horizontal FL on MLP, Membership Inference Attacks (MIA) using DP, MIA using SAM, PATE, Vertical FL. | 9 | [Heart Disease] [Credit Card Fraud] [Breaset Cancer Data] [TCGA] [CIFAR10][cifar10-pet] [Home Credit Default Risk] [Yelp] [Airbnb]|
| [SSGVQAP][ssgvap-repo] | This repository contains code for the paper [A Smart System to Generate and Validate Question Answer Pairs for COVID-19 Literature][ssgvap-paper] which was accepted ibn ACL'20. Authors are Rohan Bhambhoria, Luna Feng, Dawn Sepehr, John Chen, Conner Cowling, Sedef Kocak, Elham Dolatabadi. | An Active Learning Strategy for Data Selection, AL-Uncertainty, AL-Clustering | 1 | [CORD-19] |
| [NeuralKernelBandits][nkb-repo] | This repository contains code for the paper [An Empirical Study of Neural Kernel Bandits][nkb-paper] which was accepted in Neurips'21. Authors are Lisicki, Michal, Arash Afkanpour, and Graham W. Taylor. | Neural tangent kernel, Conjugate kernel, NNGP, Deep ensembles, Randomized Priors, NTKGP, Upper Confidence Bounds (UCB), Thompson Sampling (TS) | 7 | [Mushroom] [Statlog] [Adult][adult-nkb] [US Census 1990] [Covertype] |
| [foodprice-forecasting][fpf-repo] | This repository replicates the experiments described on pages 16 and 17 of the [2022 Edition of Canada's Food Price Report][fpf-paper]. | Time series forecasting using Prophet,  Time series forecasting using Neural prophet, Interpretable time series forecasting using N-BEATS, Ensemble of the above methods. | 3 | [FRED Economic Data] |
--------

[//]: # (Reference links for Github repositories)
[lafter-repo]: https://github.com/VectorInstitute/laftr
[god-repo]: https://github.com/VectorInstitute/gram-ood-detection
[cvp-repo]: https://github.com/VectorInstitute/Computer_Vision_Project
[pet-repo]: https://github.com/VectorInstitute/PETs-Bootcamp
[ssgvap-repo]: https://github.com/VectorInstitute/SSGVQAP
[nkb-repo]: https://github.com/VectorInstitute/NeuralKernelBandits
[fpf-repo]: https://github.com/VectorInstitute/foodprice-forecasting

[//]: # (Reference links for Research papers)
[laftr-paper]: https://arxiv.org/abs/1802.06309
[god-paper]: http://proceedings.mlr.press/v119/sastry20a.html
[ssgvap-paper]: https://aclanthology.org/2020.sdp-1.4/
[nkb-paper]: https://arxiv.org/abs/2111.03543
[fpf-paper]: https://www.dal.ca/sites/agri-food/research/canada-s-food-price-report-2022.html

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