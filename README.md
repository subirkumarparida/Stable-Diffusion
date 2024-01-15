# Stable-Diffusion

## Error resolving Stable Diffusion Jupyter Notebook run:
---------------------------------------------------------
### 1) No such file or directory: '/workspace/SD/configs/latent-diffusion/cin256-v2.yaml'
	sys.path.append('latent-diffusion')
	
	def get_model():
    		config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/cin256-v2.yaml")  
    		model = load_model_from_config(config, "latent-diffusion/models/ldm/cin256-v2/model.ckpt")
    		return model

### 2) No module named 'pytorch_lightning.utilities.distributed'
	from pytorch_lightning.utilities import rank_zero_only

### 3) ModuleNotFoundError: No module named 'clip'	
	pip install git+https://github.com/openai/CLIP.git

### 4) ModuleNotFoundError: No module named 'kornia'
	pip install kornia

### Command to generate unconditional faces:
```
CUDA_VISIBLE_DEVICES=0,1,3 python scripts/sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l 'models/ldm/' -n 3000 --batch_size 1 -c 20 -e 0.0
```
