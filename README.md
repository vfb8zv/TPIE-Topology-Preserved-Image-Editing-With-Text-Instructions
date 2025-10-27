# TPIE: Topology-Preserved Image Editing With Text Instructions

This is the official repository for the paper "TPIE: Topology-Preserved Image Editing With Text Instructions"

## Abstract:
Preserving topological structures is important in real-world applications, particularly in sensitive domains such as healthcare and medicine, where the correctness of human anatomy is critical. However, most existing image editing models focus on manipulating intensity and texture features, often overlooking object geometry within images. To address this issue, this paper introduces a novel method, Topology-Preserved Image Editing with text instructions (TPIE), that for the first time ensures the topology and geometry remaining intact in edited images through text-guided generative diffusion models. More specifically, our method treats newly generated samples as deformable variations of a given input template, allowing for controllable and structure-preserving edits. Our proposed TPIE framework consists of two key modules: (i) an autoencoder-based registration network that learns latent representations of object transformations, parameterized by velocity fields, from pairwise training images; and (ii) a novel latent conditional geometric diffusion (LCDG) model efficiently capturing the data distribution of learned transformation features conditioned on custom-defined text instructions. We validate TPIE on a diverse set of 2D and 3D images and compare them with state-of-the-art image editing approaches. Experimental results show that our method outperforms other baselines in generating more realistic images with well-preserved topology.

![TPIE Network](media/TPIE_model.jpg)

## Demo:
![TPIE Demo](media/TPIE-demo.gif)

## Repository Details:

### File Breakdown:

- `'train.ipynb' for training the velocity representation learning and latent conditional geometric diffusion models.`
- `'testing.ipynb' for testing both the models.`
- ` The directory 'TPIE' contains the model's main definition and utility functions. `
- `'/plants/labels' is a sample dataset for RGB plants.`
Please follow the dataset's format provided in '/plants/labels' to adapt TPIE for your own task. 

### Environmental Setup:
- `datasets`                     2.16.1
- `diffusers`                    0.25.1
- `einops`                       0.7.0
- `huggingface-hub`              0.20.2
- `imageio`                      2.31.4
- `neurite`                      0.2
- `nibabel`                      5.2.0
- `numpy`                        1.26.3
- `open-clip-torch`              2.23.0
- `opencv-python`                4.8.1.78
- `pandas`                       2.1.3
- `Pillow`                       10.0.1
- `scikit-image`                 0.22.0
- `scikit-learn`                 1.3.2
- `scipy`                        1.11.1
- `SimpleITK`                    2.3.1
- `torch`                        2.1.0+cu118
- `torchmetrics`                 0.11.0
- `torchvision`                  0.16.0+cu118
- `transformers`                 4.36.2



### This code was inspired by the following repositories:
- `InstructPix2Pix: https://github.com/timothybrooks/instruct-pix2pix`
- `Guided Diffusion: https://github.com/openai/guided-diffusion`
- `VoxelMorph: https://github.com/voxelmorph/voxelmorph`

