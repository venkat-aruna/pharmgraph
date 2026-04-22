# PharmGraph  
A Pharmacogenomics Knowledge Graph & Clinical Analytics Platform  

## DESCRIPTION  

PharmGraph is an interactive pharmacogenomics (PGx) knowledge graph and analytics platform designed to unify drug–gene–variant–disease relationships into a single, queryable system. It supports precision medicine by enabling users to explore how genetic variation influences drug response across different populations.

Traditional pharmacogenomics resources store information across separate databases, such as variant repositories, drug–gene interaction datasets, and clinical guideline systems. This fragmentation makes integrated analysis difficult. PharmGraph addresses this limitation by combining multiple biomedical datasets into a unified heterogeneous knowledge graph.

The system integrates data from ClinVar (genetic variants), PharmGKB/ClinPGx (clinical annotations), DGIdb (drug–gene interactions).

PharmGraph enables:
- Population-aware insights into variants and genes
- Graph-based exploration of drug–gene–disease relationships
- Prediction of missing drug–gene interactions using machine learning

Overall, PharmGraph functions as both an exploratory tool and a predictive system for discovering clinically relevant relationships beyond curated databases.

## PRERQUISITES

| Tool   | Version (recommended) |
|--------|------------------------|
| Git    | 2.30+                  |
| Python | 3.8 or higher          |


## INSTALLATION  

### 1. Clone the repository
```bash
git clone https://github.com/venkat-aruna/pharmgraph.git
python -m http.server 8000
```

### 2. Local host website
Go to your local browser and and tye the url (http://localhost:8000/)

## NOTES

PharmGraph is a research-oriented system for integrating pharmacogenomic knowledge into a unified graph-based framework. It supports exploration, interpretation, and prediction of drug–gene relationships.

## Future improvements

Expanding population coverage for underrepresented groups
Improving ML prediction accuracy and expand to edges other than genes-drug 
Enhancing clinical decision-support capabilities
Increasing dataset completeness and graph connectivity
