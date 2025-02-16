# torch-flow-models

PyTorch-implementations of Flow Models for toy data

## Usage

Install the package.

```bash
git clone https://github.com/revsic/torch-flow-models
cd torch-flow-models && pip install -e .
```

Here is the sample code[[samples/ddpm.ipynb](./samples/ddpm.ipynb)]:

```py
import torch.nn as nn

from flowmodels import DDPM, DDIMScheduler


model = DDPM(nn.Sequential(...), DDIMScheduler())

# update
optim = torch.optim.Adam(model.parameters(), LR)
for i in range(TRAIN_STEPS):
    optim.zero_grad()
    model.loss(batch).backward()
    optim.step()

# sample
sampled, trajectory = model.sample(torch.randn(...))
```

## Implemented Models

- DDPM[[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]: Denoising Diffusion Probabilistic Models, Ho et al., 2020.
    - Imports: `DDPM`, `DDPMScheduler`, `DDPMSampler`
    - Examples: [samples/ddpm.ipynb](./samples/ddpm.ipynb)
- DDIM[[arXiv:2010.02502](https://arxiv.org/abs/2010.02502)]: Denoising Diffusion Implicit Models, Song et al., 2020.
    - Imports: `DDIMScheduler`, `DDIMSampler`
    - Examples: [samples/ddpm.ipynb](./samples/ddpm.ipynb), 4. Test the model
- NCSN[[arXiv:1907.05600](https://arxiv.org/abs/1907.05600)]: Generative Modeling by Estimating Gradients of the Data Distribution, Song & Ermon, 2019.
    - Imports: `NCSN`, `NCSNScheduler`, `AnnealedLangevinDynamicsSampler`
    - Examples: [samples/ncsn.ipynb](./samples/ncsn.ipynb)
- VPSDE, VESDE[[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)]: Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020.
    - Imports: `VPSDE`, `VPSDEAncestralSampler`, `VPSDEScheduler`
    - Imports: `VESDE`, `VESDEAncestralSampler`, `VESDEScheduler`
    - Examples: TBD
- PF-ODE[[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)]: Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020.
    - Imports: `ProbabilityFlowODESampler`
    - Examples: [samples/ddpm.ipynb](./samples/ddpm.ipynb), 4. Test the model
- Rectified Flow[[arXiv:2209.03003](https://arxiv.org/abs/2209.03003)]: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022.
    - Imports: `RectifiedFlow`, `VanillaEulerSolver`
    - Examples: TBD
- InstaFlow[[arXiv:2309.06380](https://arxiv.org/abs/2309.06380)]: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, Liu et al., 2023.
    - Imports: `InstaFlow`
    - Examples: TBD
