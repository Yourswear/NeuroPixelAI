import numpy as np
from tifffile import imread, TiffWriter, TiffFile
from cellpose import utils, models, io, core, version_str, default_ops

# Image information loading
Image=imread(r'F:\SynologyDrive\LAB-share\SynologyDrive\Member-HuJiaHao\fluorescence_corsstalk_correction\file_00006.tif')
if np.min(Image)>32768:
    Image = Image-32768
Image = Image.astype(np.float32)
Std_Image = np.std(Image, axis=2)

# Parameter settings
ops = default_ops()

model = models.Cellpose(gpu=ops.gpu, device=ops.device, model_type=ops.model_type,
                        net_avg=ops.netavg)

out = model.eval(Std_Image, channels=ops.channels, diameter=ops.diameter,
                 do_3D=ops.do_3D, net_avg=ops.net_avg,
                 augment=ops.augment,
                 resample=ops.resample,
                 flow_threshold=ops.flow_threshold,
                 cellprob_threshold=ops.cellprob_threshold,
                 stitch_threshold=ops.stitch_threshold,
                 min_size=ops.min_size,
                 invert=ops.invert,
                 batch_size=ops.batch_size,
                 interp=ops.interp,
                 normalize=ops.normalize,
                 channel_axis=ops.channel_axis,
                 z_axis=ops.z_axis,
                 anisotropy=ops.anisotropy,
                 model_loaded=True)
masks, flows = out[:2]