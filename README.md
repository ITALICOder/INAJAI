# AI Project Portfolio: End-of-2024 Collection(The INAJAI name is there cause IneedAJobAI)=

Welcome to my AI Project Portfolio, showcasing a curated selection of AI-driven experiments, applications, and prototypes developed between October and December 2024. This repository highlights my journey exploring machine learning, deep learning, natural language processing, and computer vision techniques to solve real-world problems.

---

## Table of Contents

1. [About this Collection](#about-this-collection)
2. [Projects Overview](#projects-overview)
3. [Getting Started](#getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
4. [Usage](#usage)

   * [Running Individual Projects](#running-individual-projects)
   * [Examples & Demos](#examples--demos)
5. [Project Structure](#project-structure)
6. [Contributing](#contributing)
7. [Roadmap](#roadmap)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## About this Collection

This repository gathers several standalone AI projects completed during the final quarter of 2024. Each project is self-contained, with its own directory, requirements, and documentation.
These projects demonstrate proficiency in data preprocessing, model training, evaluation, and deployment, as well as experience with cloud services, containerization, and CI/CD workflows.

---
Research Details
During our investigation, we identified several intrinsic limitations of large language models (LLMs) that can compromise data integrity and model performance:

Inconsistent Scoring Scales
LLMs do not inherently verify that all training inputs adhere to a uniform scoring rubric. For example, when combining wine ratings sourced from multiple wine‐and‐food review platforms, discrepancies in score scales (e.g., 0–20 vs. 0–100) can introduce significant noise, leading the model to produce erratic or “corrupted” outputs whenever it encounters mixed-scale data.

Dataset Contamination from Irrelevant Domains
During our competitor analysis, we discovered that many of the top search results for targeted domains pointed not to established review sites but to generic domain‐resale or parking pages. These irrelevant entries comprised a substantial portion of our crawled dataset, frequently surfacing in queries and thereby skewing the model’s notion of authoritative sources.

Impact of Distributional Shifts and Concept Drift
LLMs assume that training and inference data share a similar distribution. However, over time, consumer terminology, rating conventions, and even tasting descriptors evolve—an effect known as concept drift. Without mechanisms for continual re‐calibration, models risk becoming less relevant or accurate as real‐world language usage changes.

Risk of Hallucinations and Overgeneralization
When faced with incomplete, inconsistent, or out‐of‐distribution inputs, LLMs may “hallucinate” plausible‐sounding but factually incorrect information. Ensuring rigorous data validation and standardized annotation schemes is essential to mitigate such overgeneralizations.

Additional AI Insights
Embedding Degradation
As models ingest heterogeneous data, vector embeddings can lose their discriminative power, making semantically distinct concepts appear artificially similar. Proper normalization and regular pruning of embedding spaces help preserve semantic fidelity.

Adversarial Data Vulnerabilities
LLMs are susceptible to adversarial inputs—carefully crafted examples that exploit model blind spots. Robustness testing, including adversarial perturbation analysis, uncovers these vulnerabilities predeployment.

Transfer Learning and Domain Adaptation
Fine‐tuning a pretrained LLM on a narrowly curated, high‐quality dataset can dramatically improve domain‐specific performance. Techniques such as LoRA (Low‐Rank Adaptation) enable efficient updates without retraining billions of parameters.

By addressing these challenges through stringent data curation, consistent annotation practices, and adaptive training strategies, we can greatly enhance the reliability and precision of LLM‐driven insights in wine review analysis and beyond.

## Getting Started

Follow these instructions to set up your local environment and run the AI projects.

### Prerequisites

* Python 3.8 or higher
* [pipenv](https://pipenv.pypa.io/) or [virtualenv](https://virtualenv.pypa.io/)
* Git
* (Optional) Docker & Docker Compose

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ITALICOder/INAJAI
   cd INAJAI
   ```

2. Create and activate a virtual environment:

   ```bash
   pipenv install --dev
   pipenv shell
   ```

   Or with `virtualenv`:

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

---

## Project Structure

```
ai-portfolio-2024/
├── Search_engine/                      # Custom little yet complex example of a Search Engine
│   └── README.md
├── Wine_Review_LLM/                    # Analysis of some dataset on Wine-review creating an AI with Wine Expertise
│   └── README.md
├── Competitors_Searcher/               # Website Competitors Search Engine
│   ├── USE_COMPETITOR_FINDER_LIB.py
│   └── README.md
├── requirements.txt
└── README.md  <-- you are here
```

---

## Roadmap

* [ ] Add detailed performance benchmarks for each model
* [ ] Integrate CI/CD pipeline with GitHub Actions
* [ ] Deploy selected projects as Dockerized microservices on AWS/GCP
* [ ] Expand documentation with video demos and tutorials

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

* [PyTorch](https://pytorch.org/) for deep learning framework
* [Hugging Face](https://huggingface.co/) for NLP models
* [FastAPI](https://fastapi.tiangolo.com/) for API development
* FAISS, sk-learn...
* Fellow open-source contributors and community tutorials

---

*Last updated: July 28, 2025*
