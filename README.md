# Introduction

HER2(Human Epidermal Growth Factor Receptor 2) titer is one of the most important prognostic factor for breast cancer. It is highly correlated to the response rate of treating patients with target therapy. However, the HER2 titer is usually not evaluted with the most prevalent staining method: hematoxylin and eosin stain but requires in-situ immunochemistry hybridization. The above immuno-staining is the gold standard to evaluate HER2 titer. The pathologists have established clear criteria for the evaluation for HER2 titer. However, the staining itself is expensive, so [Roche diagnostics](https://www.roche.com/about/business/diagnostics.htm) is searching for a solution using deep learning method to conclude HER2 titer directly from normal hematoxylin and eosin stain, without using the expensive immunostaining.

<img src="https://cancerres.aacrjournals.org/content/canres/72/21/5625/F1.large.jpg">

The [HEROHE challenge](https://ecdp2020.grand-challenge.org/), sponsored by Roche diganostics, was held on 2020 to put this question on the spotlight.

# Dataset Description

The training dataset contains 360 cases, 144 positives and 216 negatives, Each whole-slide image was saved in the  MIRAX format. Two whole slide images datasets of invasive breast cancer tissue samples were compiled to be used as training (360 cases) and test (150 cases) datasets. Considering the IHC HER2 scores, the dataset had 43 (12%) cases scored of 0, 47 (13%) scored of 1+, 230 (64%) score of 2+, and 39 (11%) scored of 3+. In cases with a score of 2+, 126 cases 
were ISH-negative and 104 ISH-positive. Hence, the dataset contained 144 HER2 positive cases (40%) and 215 HER2 negative cases (60%).

# Evaluation Method

They use F1 score to evaluate the performance of the models.

# Our Method

We first divided the images into 512\*512 image patches and use averaged saturation to threshold the foreground patches. Then we adopted a composite-patch method to sample 25 foreground patches from the foreground patches and concatenate them to become a large image. Next we try different model structures such as ResNet, ResNext, ResNest, EfficientNet, DenseNet to train the model. We design our output layer to be a 3 by 1 tensor, corresponding to the grading of 0, 1+, 2+ and 3+. We designed the groundtruth label as 0(000), 1+(001), 2+(011) and 3+(111), and use cross entropy to evaluate the loss. The advantage by doing so is that predicting a 0 will suffer more punishment than prediting a 2+ if ground truth label is 3+.
