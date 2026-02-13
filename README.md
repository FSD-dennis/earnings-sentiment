# earnings-sentiment
Project Background
In 2025 and 2026, multimodal LLMs that simultaneously process text, structured data, and price series have become a core innovation in predictive analytics. Financial firms now explore hybrid models that combine fundamentals (10-K, 10-Q), management sentiment, and pure market microstructure signals. You are required to design an LLM-driven multimodal fusion engine predicting post-earnings drift or short-term returns around earnings release dates.

Project Mission
Your mission is to gather earnings-related textual data, extract semantic sentiment and management tone, engineer price-based features reflecting volatility and liquidity states, and fuse these modalities into a unified ML or LLM-driven prediction model. You must then evaluate predictive power around earnings announcements for selected large-cap tickers.

Project Significance
This work reflects a global shift in 2025 toward multimodal financial models that combine narrative signals and numerical signals. Investors seek models that mimic human-level reasoning but maintain ML-level consistency and speed. The project will act as a prototype for future multimodal engines integrated into our analytics dashboard.

Detailed Requirements
- You need to retrieve recent earnings call transcripts and earnings summary text from publicly available sources.
- You need to use a modern LLM for extracting sentiment, uncertainty, confidence tone, forward-guidance strength, and management emotional polarity.
- You need to structure numerical features such as pre-earnings volatility compression, realized volatility, high-frequency liquidity gaps, and price momentum.
- You need to design a multimodal fusion model using either transformer-based architectures or embedding concatenation.
- You need to evaluate prediction accuracy on a short time window such as the 1-day, 3-day, or 7-day post-earnings return.
- You need to summarize model behaviors, feature importance, failure cases, and potential improvements.

More Computation and Infrastructure Requirements
Please do not assume access to any of our company's internal LLM API, GPU, or company server environment for this project. Please design this as a self-contained, reproducible prototype that can run on a normal local machine.

Here are the concrete expectations:

1. LLM / model choice
For this first delivery, you should:
- Use either
  a. A lightweight open-source model that can run locally (small transformer, distilled model), or
  b. An API-based LLM only if it is free-tier or easily swappable (clearly abstracted in code).
- Do not rely on long-context, large models that require GPUs.
- It is perfectly acceptable to truncate transcripts, summarize first, or work on selected sections (e.g. management discussion, guidance paragraphs).

2. Sentiment & tone extraction
You do not need a perfect LLM solution.
Start with a baseline (rule-based / lexicon / small NLP model) for:
- sentiment
- uncertainty
- confidence / cautious language
Optionally show how an LLM could replace this module later (pseudo-code or interface design is enough).

3. Compute constraints
 Assume:
- No internal GPU
- No private endpoints
- Local laptop environment only

Just in case you can't find solid free sources of data that can conveniently help you with completion, feel free to also try the following steps:

1. Lock the scope and universe today
Pick 5 to 8 large cap tickers, then pick a fixed number of recent earnings events per ticker, e.g., last 4 quarters. Also decide your prediction target, 1 day and 3 day post earnings returns are enough for v1.
2. Get the minimum viable data pipeline working first
For each earnings event, collect: transcript text or earnings summary text from free sources, then daily price data around the event window, e.g., minus 20 to plus 10 trading days. Save everything to a clean local folder structure so it is reproducible.
3. Build a baseline text signal module that runs locally
Start with a free approach like a small sentiment model or lexicon method. Extract a small set of features only, e.g., sentiment score, uncertainty score, forward guidance strength proxies like “will”, “expect”, “guidance”, and hedging words.
4. Build a baseline price feature module
Create a compact feature set: pre earnings momentum, pre earnings realized vol, post earnings gap, volume spike proxy, and a simple liquidity proxy if you can get it from daily data.
5. Fuse and evaluate
Start with a simple model, e.g., logistic regression for direction or ridge regression for returns. Use a time split, not random split. Report accuracy and a basic confusion matrix or error stats.
6. Deliverables checklist
Notebook with plots, a .py module with functions, and a short written summary describing what worked, what failed, and what you would improve next.


## env 
conda create -n finlang python=3.11 -y
conda activate finlang
pip install -r requirements.txt


