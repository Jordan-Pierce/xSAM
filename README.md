# xSAM

In-house version of EdgeSAM, MobileSAM, and SAM modules combined in the same API (to make life easier).

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 
Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision 
dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install xSAM:

```bash
pip install git+https://github.com/Jordan-Pierce/xSAM.git
```

## Getting Started
The xSAM can be loaded in the following ways:

```python
from x_segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
model_type = "vit_b"
model_type = "vit_l"
model_type = "vit_h"
model_type = "edge_sam"

sam_checkpoint = "model_weight.pt*"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```python
from mobile_sam import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(<your_image>)
```

### Aknowledgements:
- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [SAM](https://github.com/facebookresearch/segment-anything)

## Disclaimer

This repository is a scientific product and is not official communication of the National 
Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA 
GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its 
use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from 
the use of this GitHub project will be governed by all applicable Federal law. Any reference to 
specific commercial products, processes, or services by service mark, trademark, manufacturer, or 
otherwise, does not constitute or imply their endorsement, recommendation or favoring by the 
Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC 
bureau, shall not be used in any manner to imply endorsement of any commercial product or activity 
by DOC or the United States Government.


## License 

Software code created by U.S. Government employees is not subject to copyright in the United States 
(17 U.S.C. §105). The United States/Department of Commerce reserve all rights to seek and obtain 
copyright protection in countries other than the United States for Software authored in its 
entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to 
Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the 
Software outside of the United States.