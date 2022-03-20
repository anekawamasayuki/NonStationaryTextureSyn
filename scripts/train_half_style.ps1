gsudo .\.venv\Scripts\python train.py `
	--dataroot .\datasets\all  `
	--phase train `
	--fineSize 128 `
	--name all_2 `
	--use_style `
	<# --no_flip #>`
	--no_lsgan `
	--padding_type replicate `
	--model half_style `
	--which_model_netG resnet_2x_6blocks `
	--which_model_netD n_layers `
	--n_layers_D 4 `
	--which_direction AtoB `
	--lambda_A 100 `
	--dataset_mode half_crop `
	--norm batch `
	--pool_size 0 `
	--resize_or_crop no `
	--niter_decay 50 `
	--niter 50 `
	--save_epoch_freq 10 `
	--batchSize 8