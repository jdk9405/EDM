<h1 align="center">EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching</h1>

<!-- Arxiv Link, Project Link -->

<p align="center">
  <a href="https://arxiv.org/abs/2502.20685"><img src="https://img.shields.io/badge/arXiv-2502.20685-b31b1b.svg" alt="arXiv"></a>
  <a href="https://jdk9405.github.io/EDM"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen" alt="Project Page"></a>
</p>

<p align="center">
    <img src="asset/edm.png" width="85%" alt="EDM">
</p>


## ✨ Abstract
>We introduce the first learning-based dense matching algorithm, termed Equirectangular Projection-Oriented Dense Kernelized Feature Matching (EDM), specifically designed for omnidirectional images. Equirectangular projection (ERP) images, with their large fields of view, are particularly suited for dense matching techniques that aim to establish comprehensive correspondences across images. However, ERP images are subject to significant distortions, which we address by leveraging the spherical camera model and geodesic flow refinement in the dense matching method. To further mitigate these distortions, we propose spherical positional embeddings based on 3D Cartesian coordinates of the feature grid. Additionally, our method incorporates bidirectional transformations between spherical and Cartesian coordinate systems during refinement, utilizing a unit sphere to improve matching performance. We demonstrate that our proposed method achieves notable performance enhancements, with improvements of +26.72 and +42.62 in AUC@5° on the Matterport3D and Stanford2D3D datasets, respectively.

## 🕹 Inference
#### Pre-trained model
The pre-trained model of EDM is available [Matterport3D](https://drive.google.com/file/d/1RFsc-MhZ4VSxe60g9cOkhMRnQjgonvNb/view?usp=drive_link).
```
python test.py --im_A_path [IMG1 DIR] --im_B_path [IMG2 DIR]
```

## 📚 BibTex
```bibtex
@article{jung2025edm,
  title={EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching},
  author={Jung, Dongki and Choi, Jaehoon and Lee, Yonghan and Jeong, Somi and Lee, Taejae and Manocha, Dinesh and Yeon, Suyong},
  journal={arXiv preprint arXiv:2502.20685},
  year={2025}
}
```
