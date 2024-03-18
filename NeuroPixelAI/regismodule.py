from .registration import register
import numpy as np
import time

def reg(raw_image,ops):
    t11 = time.time()
    print("----------- REGISTRATION")
    n_frames, Ly, Lx = raw_image.shape
    ops['batch_size'] = n_frames
    Midrefimage = register.compute_reference(raw_image, ops)
    maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = register.compute_reference_masks(
        Midrefimage, ops)
    refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks]
    reg_image, ymax, xmax, cmax, ymax1, xmax1, cmax1, nonsense = register.register_frames(refAndMasks, raw_image,
                                                                                          rmin=-np.inf, rmax=np.inf,
                                                                                          bidiphase=ops['bidi_corrected'],
                                                                                          ops=ops, nZ=1)
    plane_times = time.time() - t11
    print("----------- Total %0.2f sec" % plane_times)
    return np.floor(reg_image).astype(np.int16)
