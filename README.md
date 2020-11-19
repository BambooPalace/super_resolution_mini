## Description
The goal of the project is to perform single image super resolution by four times on a mini DIV2K image dataset (500 training and 80 validation pairs of images), and the metric for assessment is PSNR on the test images. 

The script `main.py` is to put alongside the [MMEditing](https://github.com/open-mmlab/mmediting) repo, and implement customized experiments with MMediting image restoration or super resolution. The restoration model used is MSRResnet and default configuration file used is [`msrresnet_x4c64b16_g1_1000k_div2k.py`](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py)

The default key parameters are: 
-	Backbone: MSRResnet of 16 residual blocks ( 1517571 parameters)
-	Batch size:  16, 
-	Loss: L1 loss, 
-	Optimizer: Adam with cosine restart policy, 

## Requirements
Install requirements of MMEditing following this [intruction](https://github.com/open-mmlab/mmediting/blob/master/docs/install.md) to run this project.

## Instructions
- Git clone the [MMEditing](https://github.com/open-mmlab/mmediting) repo, download `main.py` and save in the same folder as MMEditing. 

Specifically, `main.py` make changes to default config file and perform below actions:
1. Data preparation: modify data path and annotate training images
2. Change config: change backbone depth, e.g. from 16 residual blocks to 20.
3. Check number of model parameters, default MSRResnet-16 blocks has 1517571 
4. Training model with customized options, for detailed refer to `Run scripts`
5. Inference trained model to restore data, trained model in `/checkpoint` folder. 

User can edit the code to train different restorer models.

**Examples**
- Train model for the 1st time. 1st time should use '--annotate' to produce annotation file

`python main.py --iter=8000 --log_eva_interval=400 --checkpoint_interval=400 --bs=16 --worker=6 --annotate`
- Resume training with customized options

`python main.py --iter=100000 --log_eva_interval=400 --checkpoint_interval=400 --bs=16 --worker=6 --work_dir='./results/b20/' --num_blocks=20 --resume`

- Inference data : Modify config, checkpoint and image path for `fake_gt.py` and  `test.py`. Generate fake 4x scaled HR images with `fake_gt.py` before using the MMediting provided `test.py` to restore custom images.

 
 

**optional arguments:**
  ```buildoutcfg
  -h, --help            show this help message and exit
  --resume              resume training from checkpoint
  --checkpoint CHECKPOINT
                        checkpoint path to resume from
  --annotate            re annotate training files if needed
  --iter ITER           no of iterations, default 100k, pretrained uses 300k
  --bs BS               samples per gpu, default 16
  --worker WORKER       default = 6
  --log_eva_interval LOG_EVA_INTERVAL
                        no of iters per log and evaluation
  --checkpoint_interval CHECKPOINT_INTERVAL
                        no of iters to save the checkpoint
  --lr_step             change lr policy from default Cosine Restart to step
  --work_dir WORK_DIR   path to save checkpoint and logs
  --num_blocks NUM_BLOCKS
                        default 16 blocks in srresnet
  --loss {L1Loss,MSELoss}
                        set loss function
  --check_param         only check no of parameters
```


## Results

The author used MSRResnet of 20 residual blocks ( 1812995 parameters) for model training. 

In MMEditing [doc](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan), the pretrained MSRResnet restorer after 300k iterations gets a test PSNR score of 28.97 evaluated with DIV2K datasets. This is a good indicator of expected PSNR score my model can get, although validation score is usually higher than the test score.

After 100k iterations, with the learning rate policy as cosine restart per 250k, the best validation PSNR is 28.93.


