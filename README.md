# Stable-Diffusion
### Train Stable Diffusion on CelebA-HQ:

##### Train on CelebA-HQ
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml -t --no-test --gpus 0,
```

##### Train on CelebA-HQ MULTI-GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml -t --no-test --gpus 0,1,2,3
```

##### Test on CelebA-HQ
```
CUDA_VISIBLE_DEVICES=0,1,3 python scripts/sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l 'models/ldm/' -n 3000 --batch_size 1 -c 20 -e 0.0
```



### Train Stable Diffusion on FairFace:
##### Train on FairFace_224
```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/fairfacebal-ldm-vq-4.yaml -t --no-test --gpus 0,
```

##### Train on FairFace_224 MULTI-GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/fairfacebal-ldm-vq-4.yaml -t --no-test --gpus 0,1,2,3
```

##### Test on FairFace_224
```
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py -r models/ldm/celeba256/model.ckpt -l 'models/ldm/' -n 3000 --batch_size 1 -c 20 -e 0.0
```


### Error resolving main.py run:
---------------------------------------------------------
#### 0) pip install -e ./taming-transformers
#### 1) modified main.py for rank_zero
#### 2) pip install pytorch-lightning==1.06
#### 3) Download vq-4 checkpointed model and unzip it
	wget -O models/first_stage_models/vq-f4/model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
	unzip -o model.zip
#### 4) pip install test-tube
#### 5) No module named 'albumentations'
	pip install albumentations
#### 6) AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType' (most likely due to a circular import)
	pip uninstall opencv-python-headless
	
	pip install opencv-python-headless==4.5.5.64
	pip install opencv-contrib-python-headless==4.5.5.64
#### 7) dataset npz files

#### 8) on_train_epoch_end error
	cd
	cd ..
	cd opt/conda/lib/python3.8/site-packages/pytorch_lightning/loops/
	vi fit_loop.py
		comment out those two lines containing error


### Error resolving Stable Diffusion Jupyter Notebook run:
---------------------------------------------------------

#### 1) No such file or directory: '/workspace/SD/configs/latent-diffusion/cin256-v2.yaml'
	sys.path.append('latent-diffusion')
	
	def get_model():
    		config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/cin256-v2.yaml")  
    		model = load_model_from_config(config, "latent-diffusion/models/ldm/cin256-v2/model.ckpt")
    		return model

#### 2) No module named 'pytorch_lightning.utilities.distributed'
	from pytorch_lightning.utilities import rank_zero_only

#### 3) ModuleNotFoundError: No module named 'clip'	
	pip install git+https://github.com/openai/CLIP.git

#### 4) ModuleNotFoundError: No module named 'kornia'
	pip install kornia
