# MultiCaRe-Fusion: Multimodal Clinical Image Classification from Case Reports

Course project for **AID 427: New Trends in Data Science** (E-JUST), submitted to **Dr. Ahmed Anter**.

**Team:** Nour Ezz (320230072), Marwan Ayman (320230061), section 2.

We classify medical images from published case reports into **27 image subtypes** (CT, MRI, H&E, ultrasound, endoscopy, ECG, ...) by fusing three kinds of information: the image, its caption, and clinical context (age, gender, case text). The study compares dimensionality-reduction, clustering, feature-selection and classification methods under a **leakage-safe, article-grouped** evaluation protocol.

## Dataset

**MultiCaRe v2.1**, open-access case reports from PubMed Central.

- Download: https://zenodo.org/records/14994046 (Zenodo DOI 10.5281/zenodo.14994046)
- License: CC BY-NC-SA 4.0
- Size: 130,791 images, 33,129 patients with images.
- The dataset is **not included** in this repository because of its size.

After downloading, extract it so that this layout exists:

```
Phase 1/
  DataSet/
    captions_and_labels.csv
    cases.parquet
    PMC1/ ... PMC9/        (image folders)
  Notebooks/
```

The notebooks find `../DataSet` automatically.

## Repository structure

| Path | Content |
|---|---|
| `Phase 0/` | Project proposal (PDF) |
| `Phase 1/Notebooks/01_EDA_and_Preprocessing_v2.ipynb` | EDA, cleaning, 27-class working subset, article-grouped split |
| `Phase 1/Notebooks/02_Feature_Extraction_v2.ipynb` | ResNet50, DINOv2, BiomedCLIP (image and text), HOG/colour, TF-IDF, tabular features (5,878 dims) |
| `Phase 1/Notebooks/03_Reduction_Clustering_Selection_v2.ipynb` | PCA/LDA/SVD, KPCA, SNE, t-SNE, UMAP, MDS, Isomap, LLE; KMeans/Ward/DBSCAN; filter, wrapper, embedded, GA and SA feature selection |
| `Phase 1/Notebooks/04_Models_and_Evaluation_v5.ipynb` | 7 classifiers x 5 feature sets, evaluation, over/under-fitting, tuning, voting, stacking, hierarchical model, ablation |

## How to run

1. Create an environment with Python 3.10+ and install: `numpy pandas scikit-learn matplotlib seaborn pillow pyarrow torch transformers open_clip_torch umap-learn xgboost`.
2. Download and extract the dataset (see above).
3. Run the notebooks **in order, 01 to 04**, with a local kernel. Each notebook writes its outputs to `DataSet/outputs/` for the next one. Notebook 02 needs a GPU for reasonable speed; the others run on CPU.
4. Random seed is 42 everywhere.

## Method summary

- **Split:** `StratifiedGroupKFold`, grouped by article, so images of one article never appear in both train and test. Fold 0 is the test set (4,315 images); folds 1-4 are used for cross-validation (17,673 training images).
- **Subset:** 27 of 40 subtypes with at least 250 images, at most 1,000 per subtype (21,988 images; imbalance reduced from 1025x to 3.5x).
- **Features:** frozen encoders, each block standardised and weighted by 1/sqrt(d), concatenated into 5,878 dimensions.

## Main results (test set, 4,315 images, 95% Wilson interval about +/-1.2 points)

| Approach | CV acc. | Test acc. |
|---|---|---|
| MLP, all features (default best) | 0.795 | 0.809 |
| Logistic regression, C=0.001 (tuned) | 0.809 | 0.815 |
| Voting ensemble (top-3 tuned) | 0.810 | 0.815 |
| Hierarchical (type then subtype) | - | 0.816 |
| LDA to 26 dims + classifier | 0.791 (RF) | 0.796 |

Macro ROC-AUC of the best default model is 0.986. Differences among the tuned, ensemble and hierarchical models are within the statistical noise. Image features carry most of the signal (BiomedCLIP alone: 0.776 accuracy); case text and age/gender add almost nothing.

## Citation and credit

Dataset: Nievas Offidani, M. A., and Delrieux, C., *MultiCaRe*, Zenodo, 2025, https://zenodo.org/records/14994046. Please respect the dataset's CC BY-NC-SA 4.0 license.
