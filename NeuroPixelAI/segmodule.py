from .cellpose import models

def cal_diam(std_image,ops):
    device, gpu = models.assign_device(use_torch=True, gpu=ops['gpu'], device=ops['device'])
    cp = models.CellposeModel(device=device, gpu=gpu,
                              pretrained_model=ops['model_type'],
                              diam_mean=30,
                              net_avg=ops['net_avg'])  # model for diameter calculation
    sz = models.SizeModel(device=device, pretrained_size=ops['pretrained_size'],
                          cp_model=cp)
    sz.model_type = ops['model_type']
    diameter, _ = sz.eval(std_image, channels=ops['channels'], channel_axis=None, invert=ops['invert'],
                          batch_size=ops['batch_size'],
                          augment=ops['augment'], tile=True, normalize=ops['normalize'])
    return diameter

def seg(std_image,ops):
    # Parameter settings
    device, gpu = models.assign_device(use_torch=True, gpu=ops['gpu'], device=ops['device'])

    model = models.CellposeModel(gpu=gpu, device=device, pretrained_model=ops['model_type'],
                            net_avg=ops['net_avg'])  # model for cellpose segmentation

    out = model.eval(std_image, channels=ops['channels'], diameter=ops['diameter'],
                     do_3D=ops['do_3D'], net_avg=ops['net_avg'],
                     augment=ops['augment'],
                     resample=ops['resample'],
                     flow_threshold=ops['flow_threshold'],
                     cellprob_threshold=ops['cellprob_threshold'],
                     stitch_threshold=ops['stitch_threshold'],
                     min_size=ops['min_size'],
                     invert=ops['invert'],
                     batch_size=ops['batch_size'],
                     interp=ops['interp'],
                     normalize=ops['normalize'],
                     channel_axis=ops['channel_axis'],
                     z_axis=ops['z_axis'],
                     anisotropy=ops['anisotropy'],
                     model_loaded=True)
    mask, flows = out[:2]
    return mask
