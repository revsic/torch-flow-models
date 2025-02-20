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
    - Examples: [samples/vpsde.ipynb](./samples/vpsde.ipynb), [samples/vesde.ipynb](./samples/vesde.ipynb)
- PF-ODE[[arXiv:2011.13456](https://arxiv.org/abs/2011.13456)]: Score-Based Generative Modeling through Stochastic Differential Equations, Song et al., 2020.
    - Imports: `ProbabilityFlowODESampler`
    - Examples: [samples/ddpm.ipynb](./samples/ddpm.ipynb), 4. Test the model
- Rectified Flow[[arXiv:2209.03003](https://arxiv.org/abs/2209.03003)]: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, Liu et al., 2022.
    - Imports: `RectifiedFlow`, `VanillaEulerSolver`
    - Examples: [samples/rf.ipynb](./samples/rf.ipynb)
- InstaFlow[[arXiv:2309.06380](https://arxiv.org/abs/2309.06380)]: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation, Liu et al., 2023.
    - Imports: `InstaFlow`
    - Examples: [samples/instaflow.ipynb](./samples/instaflow.ipynb)
- Shortcut Model[[arXiv:2410.12557](https://arxiv.org/abs/2410.12557)]: One Step Diffusion via Shortcut Models, Frans et al., 2024.
    - Imports: `ShortcutModel`, `ShortcutEulerSolver`
    - Examples: [samples/shortcut.ipynb](./samples/shortcut.ipynb)
- Rectified Diffusion[[arXiv:2410.07303](https://arxiv.org/abs/2410.07303)]: Straightness Is Not Your Need in Rectified Flow, Wang et al., 2024.
    - Imports: `RectifiedDiffusion`
    - Examples: [samples/rd.ipynb](./samples/rd.ipynb)
- Consistency Models[[arXiv:2303.01469](https://arxiv.org/abs/2303.01469)], Song et al., 2023.
    - Imports: `ConsistencyModel`, `MultistepConsistencySampler`
    - Examples: [samples/cm.ipynb](./samples/cm.ipynb)
- Consistency Flow Matching[[arXiv:2407.02398](https://arxiv.org/abs/2407.02398)]: Defining Straight Flows with Velocity Consistency, Yang et al., 2024.
    - Imports: `ConsistencyFlowMatching`
    - Examples: [samples/consistencyfm.ipynb](./samples/consistencyfm.ipynb)
- sCT[[arXiv:2410.11081](https://arxiv.org/abs/2410.11081)]: Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models, Lu & Song, 2024.
    - Imports: `ScaledContinuousCM`, `ScaledContinuousCMScheduler`
    - Examples: [samples/sct.ipynb](./samples/sct.ipynb)
- DSBM[[arXiv:2303.16852](https://arxiv.org/abs/2303.16852)]: Diffusion Schrodinger Bridge Matching, Shi et al., 2023.
    - Imports: `DiffusionSchrodingerBridgeMatching`, `ModifiedVanillaEulerSolver`
    - Examples: [samples/dsbm.ipynb](./samples/dsbm.ipynb)
- FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing, Deng et al., 2024. 
    - Imports: `FireFlowSolver`, `FireFlowInversion`
    - Examples: [samples/rf.ipynb], 4. Test the model
