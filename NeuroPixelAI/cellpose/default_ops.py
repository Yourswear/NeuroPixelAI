def seg_default_ops():
    """ default options to run pipeline """
    chan1 = 1
    chan2 = 0
    diameter = 21.5
    return {
        'gpu': True ,
        'device': '0',  # which gpu device to use, use an integer for torch, or mps for M1, (default: '0')
        'model_type': './cellpose/model/cyto',
        'pretrained_size': './cellpose/size_model/size_cytotorch_0.npy',
        'channels': [chan1, chan2],
        'diameter': diameter,
        'do_3D': False,
        'net_avg': True,
        'augment': False,
        'resample': True,
        'flow_threshold': 0.4,
        'cellprob_threshold': 0,
        'stitch_threshold': 0,
        'min_size': 15,  # minimum number of pixels per mask, can turn off with -1, (default: 15)
        'invert': False,  # invert grayscale channel
        'batch_size': 8,
        'interp': False,  # interpolate when running dynamics (default: False)
        'normalize': True,
        'channel_axis': None,  # axis of image which corresponds to image channels (default: None)
        'z_axis': None,
        'anisotropy': 1.0,  # anisotropy of volume in 3D (default: 1.0)
        'model_loaded': True
    }