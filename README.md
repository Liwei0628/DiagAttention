# Diagonalized Attention for Individualized Regression: Latent-Row Localization and Prediction

A PyTorch implementation of diagonalized attention for individualized regression in language and vision applications.

## 📂 Repository Structure

The repository consists of three main components:

- **`classAttention.py`**: Implements the single-head (`DiagAttRegression`) and multi-head (`MultiDiagAttRegression`) diagonalized attention models, along with training and prediction functions (`RunRegression`).
- **`utils.py`**: Provides utility functions and classes for data preprocessing, dataset handling, early stopping, and latent-row localization.
- **`example/`**: Demonstrates the application of diagonalized attention to real-world data.

## 📖 Citation

If you find our work useful, please cite our paper:

📄 **Paper:** [Diagonalized Attention for Individualized Regression: Latent-Row Localization and Prediction](https://arxiv.org/abs/2609.21320)

```bibtex
@article{peng2026diagonalized,
  title={Diagonalized Attention for Individualized Regression: Latent-Row Localization and Prediction},
  author={Peng, Borui and Lin, Liwei and Wang, Feifei and Feng, Long},
  journal={arXiv preprint arXiv:2609.21320},
  year={2026}
}
```
