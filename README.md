# Implemented Papers

Welcome to the **Implemented Papers** repository! This is where I re-create research papers in deep learning and other numerical fields. I'm doing this to improve my understanding of the latest techniques and to build a handy collection of reusable tools and code.

## Overview

**Implemented Papers** is designed to translate innovative research papers into functional code. By re-implementing these papers, I aim to:

- Enhance my understanding of advanced deep learning techniques.
- Create modular, reusable code that serves as both a learning tool and a practical resource.
- Build a personal framework with classes and utilities that reflect modern research.

*Note*: Although I'm coding everything from scratch to deepen my understanding and experience, I occasionally use coding assistants such as ChatGPT and GitHub Copilot. These tools help ensure that the documentation for functions in the framework remains consistent and well-organized.

## Features

- **Re-Implementations**: Code for various deep learning papers and numerical methods.
- **Modular Framework**: A growing collection of classes and functions that can be used across different projects.
- **Examples**: Example scripts demonstrating the use of implemented methods.

## Implemented Papers

Here is a list of the papers I've implemented, with links to their respective directories:

1. **[NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION](papers/nice/NICE.md)**, *Laurent Dinh, David Krueger, Yoshua Bengio* ; 2014; [arXiv](https://arxiv.org/pdf/1410.8516) 
> NICE proposes a framework for learning complex high-dimensional densities using invertible transformations. It maps data to a latent space with independent variables, ensuring easy computation of the Jacobian and its inverse. This approach, based on deep neural networks, allows efficient generative modeling with strong results on image datasets.

> I re-implemented the NICE model and reproduced the experiments on the MNIST dataset. Additionally, I conducted a personal experiment using a simple sinusoidal 2D distribution as a proof of concept, demonstrating the model's effectiveness in a custom scenario.

2.  **[DENSITY ESTIMATION USING REAL NVP](papers/real_nvp/real%20NVP.md)**, *Laurent Dinh*, *Jascha Sohl-Dickstein*, *Samy Bengio*; 2017; [arXiv](https://arxiv.org/pdf/1605.08803)
> RealNVP (Real-valued Non-Volume Preserving transformations) builds upon the invertible transformation framework introduced in NICE. By using affine coupling layers, RealNVP allows for the exact computation of log-determinants of the Jacobian, facilitating scalable density estimation for high-dimensional data. The model is especially successful in generative modeling tasks on image datasets, producing high-quality samples with minimal computational overhead.

> I re-implemented the RealNVP model and reproduced the experiment on the CelebA dataset. Similar to the NICE paper, I conducted a personal experiment using a sinusoidal 2D distribution to further demonstrate RealNVP's capacity in modeling complex distributions effectively. 


3. **[GLOW: GENERATIVE FLOW WITH INVERTIBLE 1x1 CONVOLUTIONS](papers/glow/GLOW.md)**, Diederik P. Kingma, Prafulla Dhariwal; 2018; [arXiv](https://arxiv.org/abs/1807.03039)

> GLOW (Generative Flow with Invertible 1x1 Convolutions) further develops the invertible flow framework by introducing invertible 1x1 convolutions. These convolutions act as a generalization of the permutation operation in previous flow models, allowing for more expressive transformations and better capturing dependencies between dimensions. GLOW achieves state-of-the-art results in density estimation and image generation, demonstrating its ability to generate high-fidelity and diverse images.

> I re-implemented the GLOW model and reproduced the experiments (conditional and unconditional) on the CIFAR-10 dataset.

... To be continued :-D

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## TODO

I plan to do some documentation for the framework. There are many papers I want to explore in this repo, here for my future self of a set of interesting papers :
- Learning to learn by gradient descent by gradient descent
- Attention Is All You Need - A lot of the building blocks are already on my computer so it should be pretty simple. Also in the [MOzART project](https://github.com/julienbrandoit/INFO8010---MOZART---Generating-Music-with-Transformers).
- Deep Residual Learning for Image Recognition
- GUIDED IMAGE GENERATION WITH CONDITIONAL INVERTIBLE NEURAL NETWORKS