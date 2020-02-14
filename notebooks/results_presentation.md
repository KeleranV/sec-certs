## Features

| Feature name | Feature type | N. unique val. | N. missing | notes |
| --- | --- | --- | --- | --- |
| category | nominal | 15 | 0 | - |
| scheme | nominal | 17 | 0 | 8 prevalent, merge below 100? |
| n_updates | ordinal | 12 | 0 | set intervals [0,0), (1,1], (2,2], (3, inf) |
| sec_level_processed | ordinal | 14 | 0 | + stands for additional defenses |
| cert_lab | nominal | 13 | 2740 | Could be very useful, need to get more |
| n_pages | numerical | - | 923 | Split into categories by histogram | 
| cert_date | Date | - | 0 | |
| archived_date | Date | - | 0 | Beware of huge number of archived in Sep. 2019|
| manufacturer | nominal | 807 | 10 | Need to merge if it should be of some use |
| protection_profiles | nominal | 216 | 0 (or 2300)| Take length of list as feature |
| pdf_encrypted | boolean | True/False | 674 | If missing set false |
| defenses | nominal | - | 668 | To be processed to number of keys |
| crypto_algs | nominal | - | 668 | To be processed to number of algs. |

## Feature correlations w.r.t. `sec_level_cat`

```
sec_level_cat            1.000000
n_crypto_algs            0.463987
n_protection_profiles    0.322093
n_defenses               0.309062
n_pages_cat              0.213559
n_updates_cat            0.189886
n_updates                0.173871
cert_year                0.167917
n_pages                  0.152911
pdf_encrypted           -0.130792
Name: sec_level_cat, dtype: float64
```

## Accuracy of Random Forest classifier

- Goal: Guess security level (EAL1 - EAL7+), 12 classes in total
- Accuracy on a test set: Roughly 65%
- Managed to train Neural Network to get to 60%

## Confusion matrix

- Goal: Describe what is misclassified as what

![alt text](confusion_matrix.png "Title")

## Feature importances

- Goal: Say how important specific features are for the Random Forest classifier

![alt text](feature_importances.png "Title")


## Random scatter plots

- Goal: visualize two variables (and plot target variable as a color), e.g. year of certification on x-axis, number of employed cryptographic algorithms on y axis:

![alt text](year_algs_scatter.png "Title")

## Principal component analysis

- Goal: Reduce number of dimensions of data with loosing as little information as possible. Plot the results. E.g. plot decision boundary of simple logistic regression (binary classification EAL1 vs. EAL7+) when features are projected to 2D

![alt text](pca.png "Title")

- Tensorboard: `tensorboard --logdir=/Users/adam/phd/projects/certificates/data_dir`
- `Total variance described: 44.7%.` means that we lose 55,3% of information when reducing from 15 to 3 dimensions (tensorboard uses 3D plots).