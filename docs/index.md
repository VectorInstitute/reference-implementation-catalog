# Vector Institute Reference Implementation Catalog

<div class="catalog-header" markdown>
Welcome to the Vector Institute Reference Implementation Catalog!
The catalog is a curated collection of high quality implementations
developed by researchers and engineers at the Vector Institute. This catalog provides
access to repositories that demonstrate state-of-the-art techniques across a wide
range of AI domains.
</div>

<div class="catalog-stats">
  <div class="stat">
    <div class="stat-number">100+</div>
<style>
.dataset-tag {
  display: inline-block;
  background-color: #6a5acd;
  color: white;
  padding: 0.1rem 0.4rem;
  border-radius: 0.8rem;
  margin-right: 0.2rem;
  margin-bottom: 0.2rem;
  font-size: 0.7rem;
  font-weight: 500;
  white-space: nowrap;
}
</style>

    <div class="stat-label">Reference Implementations</div>
  </div>
  <div class="stat">
    <div class="stat-number">7</div>
    <div class="stat-label">Years of Research</div>
  </div>
</div>

## Browse Implementations by Year



















=== "2024"

    <div class="grid cards" markdown>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/rag" title="Go to Repository">RAG</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains demos for various Retrieval Augmented Generation techniques using different libraries.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Cloud search via LlamaHub">Cloud search via LlamaHub</span>
        <span class="tag" data-tippy="Document search via LangChain">Document search via LangChain</span>
        <span class="tag" data-tippy="LlamaIndex for OpenAI and Cohere models">LlamaIndex for OpenAI and Cohere models</span>
        <span class="tag" data-tippy="Hybrid Search via Weaviate Vector Store">Hybrid Search via Weaviate Vector Store</span>
        <span class="tag" data-tippy="Evaluation via RAGAS library">Evaluation via RAGAS library</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Vectors 2021 Annual Report</span>  <span class="dataset-tag">PubMed Doc</span>  <span class="dataset-tag">Banking Deposits</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/fa" title="Go to Repository">Finetuning and Alignment</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains demos for finetuning techniques for LLMs focussed on reducing computational cost.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="DDP">DDP</span>
        <span class="tag" data-tippy="FSDP">FSDP</span>
        <span class="tag" data-tippy="Instruction Tuning">Instruction Tuning</span>
        <span class="tag" data-tippy="LoRA">LoRA</span>
        <span class="tag" data-tippy="DoRA">DoRA</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">samsam</span>  <span class="dataset-tag">imdb</span>  <span class="dataset-tag">Bias-DeBiased</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/pe-lab" title="Go to Repository">Prompt Engineering Laboratory</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains demos for various Prompt Engineering techniques, along with examples for Bias quantification, text classification.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Stereotypical Bias Analysis">Stereotypical Bias Analysis</span>
        <span class="tag" data-tippy="Sentiment inference">Sentiment inference</span>
        <span class="tag" data-tippy="Finetuning using HF Library">Finetuning using HF Library</span>
        <span class="tag" data-tippy="Activation Generation">Activation Generation</span>
        <span class="tag" data-tippy="Train and Test Model for Activations without Prompts">Train and Test Model for Activations without Prompts</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Crows-pairs</span> <span class="dataset-tag">crow-pairs-pe-lab</span>  <span class="dataset-tag">sst5</span> <span class="dataset-tag">sst5-pe-lab</span>  <span class="dataset-tag">czarnowska templates</span> <span class="dataset-tag">czar-templ-pe-lab</span>  <span class="dataset-tag">cnn_dailymail</span>  <span class="dataset-tag">ag_news</span>  <span class="dataset-tag">Weather and sports data</span>  <span class="dataset-tag">Other</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/bmu" title="Go to Repository">bias-mitigation-unlearning</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains code for the paper [Can Machine Unlearning Reduce Social Bias in Language Models?][bmu-paper] which was published at EMNLP'24 in the Industry track. <br>Authors are Omkar Dige, Diljot Arneja, Tsz Fung Yau, Qixuan Zhang, Mohammad Bolandraftar, Xiaodan Zhu, Faiza Khan Khattak.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="PCGU">PCGU</span>
        <span class="tag" data-tippy="Task vectors and DPO for Machine Unlearning">Task vectors and DPO for Machine Unlearning</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">BBQ</span> <span class="dataset-tag">bbq-bmu</span>  <span class="dataset-tag">Stereoset</span> <span class="dataset-tag">stereoset-bmu</span>  <span class="dataset-tag">Link1</span> <span class="dataset-tag">link1-bmu</span>  <span class="dataset-tag">Link2</span> <span class="dataset-tag">link2-bmu</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/cyclops" title="Go to Repository">cyclops-workshop</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains demos for using [CyclOps] package for clinical ML evaluation and monitoring.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="XGBoost">XGBoost</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Diabetes 130-US hospitals dataset for years 1999-2008</span> <span class="dataset-tag">diabetes-cyclops</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/odyssey" title="Go to Repository">odyssey</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This is a library created with research done for the paper [EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records][odyssey-paper] published at ArXiv'24. <br>Authors are Adibvafa Fallahpour, Mahshid Alinoori, Wenqian Ye, Xu Cao, Arash Afkanpour, Amrit Krishnan.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="EHRMamba">EHRMamba</span>
        <span class="tag" data-tippy="XGBoost">XGBoost</span>
        <span class="tag" data-tippy="Bi-LSTM">Bi-LSTM</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">MIMIC-IV</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/diffusion" title="Go to Repository">Diffusion model bootcamp</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains demos for various diffusion models for tabular and time series data.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="TabDDPM">TabDDPM</span>
        <span class="tag" data-tippy="TabSyn">TabSyn</span>
        <span class="tag" data-tippy="ClavaDDPM">ClavaDDPM</span>
        <span class="tag" data-tippy="CSDI">CSDI</span>
        <span class="tag" data-tippy="TSDiff">TSDiff</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Physionet Challenge 2012</span>  <span class="dataset-tag">wiki2000</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/nmb" title="Go to Repository">News Media Bias</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>This repository contains code for libraries and experiments to recognise and evaluate bias and fakeness within news media articles via LLMs.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Bias evaluation via LLMs">Bias evaluation via LLMs</span>
        <span class="tag" data-tippy="finetuning and data annotation via LLM for fake news detection">finetuning and data annotation via LLM for fake news detection</span>
        <span class="tag" data-tippy="Supervised finetuning for debiasing sentence">Supervised finetuning for debiasing sentence</span>
        <span class="tag" data-tippy="NER for biased phrases via LLMS">NER for biased phrases via LLMS</span>
        <span class="tag" data-tippy="Evaluate using DeepEval library">Evaluate using DeepEval library</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">News Media Bias Full data</span> <span class="dataset-tag">nmb-data</span>  <span class="dataset-tag">Toxigen</span>  <span class="dataset-tag">Nela GT</span>  <span class="dataset-tag">Debiaser data</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/nmb-plus" title="Go to Repository">News Media Bias Plus</a></h3>
        <span class="tag year-tag">2024</span>
    </div>
    <p>Continuation of News Media Bias project, this repository contains code for libraries and experiments to collect and annotate data, recognise and evaluate bias and fakeness within news media articles via LLMs and LVMs.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Bias evaluation via LLMs and VLMs">Bias evaluation via LLMs and VLMs</span>
        <span class="tag" data-tippy="finetuning and data annotation via LLM for fake news detection">finetuning and data annotation via LLM for fake news detection</span>
        <span class="tag" data-tippy="supervised finetuning for debiasing sentence">supervised finetuning for debiasing sentence</span>
        <span class="tag" data-tippy="NER for biased entities via LLMS">NER for biased entities via LLMS</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">News Media Bias Plus Full Data</span> <span class="dataset-tag">nmb-plus-full-data</span>  <span class="dataset-tag">NMB Plus Named Entities</span> <span class="dataset-tag">nmb-plus-entities</span>
    </div>
    </div>

    </div>

=== "2023"

    <div class="grid cards" markdown>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/anomaly" title="Go to Repository">Anomaly Detection Project</a></h3>
        <span class="tag year-tag">2023</span>
    </div>
    <p>This repository contains demos for various supervised and unsupervised anomaly detection techniques in domains such as Fraud Detection, Network Intrusion Detection, System Monitoring and image, Video Analysis.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="AMNet">AMNet</span>
        <span class="tag" data-tippy="GCN">GCN</span>
        <span class="tag" data-tippy="SAGE">SAGE</span>
        <span class="tag" data-tippy="OCGNN">OCGNN</span>
        <span class="tag" data-tippy="DON">DON</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">On Vector Cluster</span> <span class="dataset-tag">cluster-anomaly</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/ssl" title="Go to Repository">SSL Bootcamp</a></h3>
        <span class="tag year-tag">2023</span>
    </div>
    <p>This repository contains demos for self-supervised techniques such as contrastive learning, masked modeling and self distillation.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Internal Contrastive Learning">Internal Contrastive Learning</span>
        <span class="tag" data-tippy="LatentOD-AD">LatentOD-AD</span>
        <span class="tag" data-tippy="TabRet">TabRet</span>
        <span class="tag" data-tippy="SimMTM">SimMTM</span>
        <span class="tag" data-tippy="Data2Vec">Data2Vec</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Beijing Air Quality</span> <span class="dataset-tag">baq-ssl</span>  <span class="dataset-tag">BRFSS</span> <span class="dataset-tag">brfss-ssl</span>  <span class="dataset-tag">Stroke Prediction</span> <span class="dataset-tag">stroke-ssl</span>  <span class="dataset-tag">STL10</span> <span class="dataset-tag">stl-10-ssl</span>  <span class="dataset-tag">Link1</span> <span class="dataset-tag">Link1-ssl</span>  <span class="dataset-tag">Link2</span> <span class="dataset-tag">Link2-ssl</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/ci-lab" title="Go to Repository">Causal Inference Lab</a></h3>
        <span class="tag year-tag">2023</span>
    </div>
    <p>This repository contains code to estimate the causal effects of an intervention on some measurable outcome primarily in the health domain.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Naive ATE">Naive ATE</span>
        <span class="tag" data-tippy="TARNet">TARNet</span>
        <span class="tag" data-tippy="DragonNet">DragonNet</span>
        <span class="tag" data-tippy="Double Machine Learning">Double Machine Learning</span>
        <span class="tag" data-tippy="T Learner">T Learner</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Infant Health and Development Program</span> <span class="dataset-tag">IHDP</span>  <span class="dataset-tag">Jobs</span>  <span class="dataset-tag">Twins</span>  <span class="dataset-tag">Berkeley admission</span>  <span class="dataset-tag">Government Census</span>  <span class="dataset-tag">Compas</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/hvaic" title="Go to Repository">HV-Ai-C</a></h3>
        <span class="tag year-tag">2023</span>
    </div>
    <p>This repository implements a Reinforcement Learning agent to optimize energy consumption within Data Centers.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="RL agents performing Random action">RL agents performing Random action</span>
        <span class="tag" data-tippy="Fixed action">Fixed action</span>
        <span class="tag" data-tippy="Q Learning">Q Learning</span>
        <span class="tag" data-tippy="Hyperspace Neighbor Penetration">Hyperspace Neighbor Penetration</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <em>No public datasets available</em>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/flex-model" title="Go to Repository">Flex Model</a></h3>
        <span class="tag year-tag">2023</span>
    </div>
    <p>This repository contains code for the paper [FlexModel: A Framework for Interpretability of Distributed Large Language Models][flex-model-paper]. <br> Authors are Matthew Choi, Muhammad Adil Asif, John Willes, David Emerson.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Distributed Interpretability">Distributed Interpretability</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <em>No public datasets available</em>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/vbll" title="Go to Repository">VBLL</a></h3>
        <span class="tag year-tag">2023</span>
    </div>
    <p>This repository contains code for the paper [Variational Bayesian Last Layers][vbll-paper]. <br> Authors are James Harrison, John Willes, Jasper Snoek.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Variational Bayesian Last Layers">Variational Bayesian Last Layers</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">MNIST</span>  <span class="dataset-tag">FashionMNIST</span>
    </div>
    </div>

    </div>

=== "2022"

    <div class="grid cards" markdown>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/recsys" title="Go to Repository">Recommendation Systems</a></h3>
        <span class="tag year-tag">2022</span>
    </div>
    <p>This repository contains demos for various RecSys techniques such as Collaborative Filtering, Knowledge Graph, RL based, Sequence Aware, Session based etc.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="SVD++">SVD++</span>
        <span class="tag" data-tippy="NeuMF">NeuMF</span>
        <span class="tag" data-tippy="Plot based">Plot based</span>
        <span class="tag" data-tippy="Two tower">Two tower</span>
        <span class="tag" data-tippy="SVD">SVD</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Amazon-recsys</span>  <span class="dataset-tag">careervillage</span>  <span class="dataset-tag">movielens-recsys</span>  <span class="dataset-tag">tmdb</span>  <span class="dataset-tag">LastFM</span>  <span class="dataset-tag">yoochoose</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/forecasting-dl" title="Go to Repository">Forecasting with Deep Learning</a></h3>
        <span class="tag year-tag">2022</span>
    </div>
    <p>This repository contains demos for a variety of forecasting techniques for Univariate and Multivariate time series, spatiotemporal forecasting etc.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Exponential Smoothing">Exponential Smoothing</span>
        <span class="tag" data-tippy="Persistence Forecasting">Persistence Forecasting</span>
        <span class="tag" data-tippy="Mean Window Forecast">Mean Window Forecast</span>
        <span class="tag" data-tippy="Prophet">Prophet</span>
        <span class="tag" data-tippy="Neuralphophet">Neuralphophet</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Canadian Weather Station Data</span>  <span class="dataset-tag">BoC Exchange rate</span>  <span class="dataset-tag">Electricity Consumption</span>  <span class="dataset-tag">Road Traffic Occupancy</span>  <span class="dataset-tag">Influenza-Like Illness Patient Ratios</span>  <span class="dataset-tag">Walmart M5 Retail Product Sales</span>  <span class="dataset-tag">WeatherBench</span>  <span class="dataset-tag">Grocery Store Sales</span>  <span class="dataset-tag">Economic Data with Food CPI</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/pe" title="Go to Repository">Prompt Engineering</a></h3>
        <span class="tag year-tag">2022</span>
    </div>
    <p>This repository contains demos for a variety of Prompt Engineering techniques such as fairness measurement via sentiment analysis, finetuning, prompt tuning, prompt ensembling etc.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Bias Quantification & Probing">Bias Quantification & Probing</span>
        <span class="tag" data-tippy="Stereotypical Bias Analysis">Stereotypical Bias Analysis</span>
        <span class="tag" data-tippy="Binary sentiment analysis task">Binary sentiment analysis task</span>
        <span class="tag" data-tippy="Finetuning using HF Library">Finetuning using HF Library</span>
        <span class="tag" data-tippy="Gradient-Search for Instruction Prefix">Gradient-Search for Instruction Prefix</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Crow-pairs</span>  <span class="dataset-tag">sst5</span>  <span class="dataset-tag">cnn_dailymail</span>  <span class="dataset-tag">ag_news</span>  <span class="dataset-tag">Tweet-data</span>  <span class="dataset-tag">Other</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/naa" title="Go to Repository">NAA</a></h3>
        <span class="tag year-tag">2022</span>
    </div>
    <p>This repository contains code for the paper [Bringing the State-of-the-Art to Customers: A Neural Agent Assistant Framework for Customer Service Support][naa-paper] published at EMNLP'22 in the industry track. <br> Authors are Stephen Obadinma, Faiza Khan Khattak, Shirley Wang, Tania Sidhorn, Elaine Lau, Sean Robertson, Jingcheng Niu, Winnie Au, Alif Munim, Karthik Raja Kalaiselvi Bhaskar.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Context Retrieval using SBERT bi-encoder">Context Retrieval using SBERT bi-encoder</span>
        <span class="tag" data-tippy="Context Retrieval using SBERT cross-encoder">Context Retrieval using SBERT cross-encoder</span>
        <span class="tag" data-tippy="Intent identification using BERT">Intent identification using BERT</span>
        <span class="tag" data-tippy="Few Shot Multi-Class Text Classification with BERT">Few Shot Multi-Class Text Classification with BERT</span>
        <span class="tag" data-tippy="Multi-Class Text Classification with BERT">Multi-Class Text Classification with BERT</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">ELI5</span>  <span class="dataset-tag">MSMARCO</span>
    </div>
    </div>

    </div>

=== "2021"

    <div class="grid cards" markdown>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/pet" title="Go to Repository">Privacy Enhancing Technologies</a></h3>
        <span class="tag year-tag">2021</span>
    </div>
    <p>This repository contains demos for Privacy, Homomorphic Encryption, Horizontal and Vertical Federated Learning, MIA, and PATE.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Vanilla SGD">Vanilla SGD</span>
        <span class="tag" data-tippy="DP SGD">DP SGD</span>
        <span class="tag" data-tippy="DP Logistic Regression">DP Logistic Regression</span>
        <span class="tag" data-tippy="Homomorphic Encryption for MLP">Homomorphic Encryption for MLP</span>
        <span class="tag" data-tippy="Horizontal FL">Horizontal FL</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">Heart Disease</span>  <span class="dataset-tag">Credit Card Fraud</span>  <span class="dataset-tag">Breaset Cancer Data</span>  <span class="dataset-tag">TCGA</span>  <span class="dataset-tag">CIFAR10</span> <span class="dataset-tag">cifar10-pet</span>  <span class="dataset-tag">Home Credit Default Risk</span>  <span class="dataset-tag">Yelp</span>  <span class="dataset-tag">Airbnb</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/ssgvap" title="Go to Repository">SSGVQAP</a></h3>
        <span class="tag year-tag">2021</span>
    </div>
    <p>This repository contains code for the paper [A Smart System to Generate and Validate Question Answer Pairs for COVID-19 Literature][ssgvap-paper] which was accepted in ACL'20. <br> Authors are Rohan Bhambhoria, Luna Feng, Dawn Sepehr, John Chen, Conner Cowling, Sedef Kocak, Elham Dolatabadi.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="An Active Learning Strategy for Data Selection">An Active Learning Strategy for Data Selection</span>
        <span class="tag" data-tippy="AL-Uncertainty">AL-Uncertainty</span>
        <span class="tag" data-tippy="AL-Clustering">AL-Clustering</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">CORD-19</span>
    </div>
    </div>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/fpf" title="Go to Repository">foodprice-forecasting</a></h3>
        <span class="tag year-tag">2021</span>
    </div>
    <p>This repository replicates the experiments described on pages 16 and 17 of the [2022 Edition of Canada's Food Price Report][fpf-paper].</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Time series forecasting using Prophet">Time series forecasting using Prophet</span>
        <span class="tag" data-tippy="Time series forecasting using Neural prophet">Time series forecasting using Neural prophet</span>
        <span class="tag" data-tippy="Interpretable time series forecasting using N-BEATS">Interpretable time series forecasting using N-BEATS</span>
        <span class="tag" data-tippy="Ensemble of the above methods">Ensemble of the above methods</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">FRED Economic Data</span>
    </div>
    </div>

    </div>

=== "2020"

    <div class="grid cards" markdown>
    <div class="card" markdown>
    <div class="header">
        <h3><a href="https://github.com/VectorInstitute/cvp" title="Go to Repository">Computer_Vision_Project</a></h3>
        <span class="tag year-tag">2020</span>
    </div>
    <p>This repository tackles different problems such as defect detection, footprint extraction, road obstacle detection, traffic incident detection, and segmentation of medical procedures.</p>
    <div class="tag-container">
        <span class="tag" data-tippy="Semantic segmentation using Unet">Semantic segmentation using Unet</span>
        <span class="tag" data-tippy="Unet++">Unet++</span>
        <span class="tag" data-tippy="FCN">FCN</span>
        <span class="tag" data-tippy="DeepLabv3">DeepLabv3</span>
        <span class="tag" data-tippy="Anomaly segmentation">Anomaly segmentation</span>
    </div>
    <div class="datasets">
        <strong>Datasets:</strong> <span class="dataset-tag">SpaceNet Building Detection V2</span>  <span class="dataset-tag">MVTEC</span>  <span class="dataset-tag">ICDAR2015</span>  <span class="dataset-tag">PASCAL_VOC</span>  <span class="dataset-tag">DOTA</span>  <span class="dataset-tag">AVA</span>  <span class="dataset-tag">UCF101-24</span>  <span class="dataset-tag">J-HMDB-21</span>
    </div>
    </div>

    </div>
