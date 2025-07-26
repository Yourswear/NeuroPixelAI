from registration import register
import numpy as np
import time

def reg(raw_image,ops):
    t11 = time.time()
    print("----------- REGISTRATION")
    n_frames, Ly, Lx = raw_image.shape
    batch_size = ops['batch_size'] = 200
    if n_frames<200:
        Midrefimage = register.compute_reference(raw_image, ops)
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = register.compute_reference_masks(
            Midrefimage, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks]
        reg_image, ymax, xmax, cmax, ymax1, xmax1, cmax1, nonsense = register.register_frames(refAndMasks, raw_image,
                                                                                              rmin=-np.inf, rmax=np.inf,
                                                                                              bidiphase=ops[
                                                                                                  'bidi_corrected'],
                                                                                              ops=ops, nZ=1)
    else:
        temp_image = raw_image[1:200,:,:]
        Midrefimage = register.compute_reference(temp_image, ops)
        del temp_image
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = register.compute_reference_masks(
            Midrefimage, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks]
        reg_image = np.zeros([n_frames, Ly, Lx])
        for i in range(n_frames//200):
            print(i)
            temp_image = raw_image[0:200,:,:]
            temp_reg_image, ymax, xmax, cmax, ymax1, xmax1, cmax1, nonsense = register.register_frames(refAndMasks, temp_image,
                                                                                                  rmin=-np.inf, rmax=np.inf,
                                                                                                  bidiphase=ops['bidi_corrected'],
                                                                                                  ops=ops, nZ=1)
            raw_image = raw_image[200:, :, :]
            reg_image[i*200:i*200+200, :, :] = temp_reg_image
        if n_frames%200 > 1:
            i = i + 1
            temp_reg_image, ymax, xmax, cmax, ymax1, xmax1, cmax1, nonsense = register.register_frames(refAndMasks,
                                                                                                       raw_image,
                                                                                                       rmin=-np.inf,
                                                                                                       rmax=np.inf,
                                                                                                       bidiphase=ops[
                                                                                                           'bidi_corrected'],
                                                                                                       ops=ops, nZ=1)
            reg_image[i * 200 :, :, :] = temp_reg_image
    plane_times = time.time() - t11
    print("----------- Total %0.2f sec" % plane_times)
    return np.floor(reg_image).astype(np.int16)
