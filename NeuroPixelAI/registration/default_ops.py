def reg_default_ops():
    """ default options to run pipeline """
    return {
        # Suite2p version
        # file input/output settings

        'look_one_level_down': False,  # whether to look in all subfolders when searching for tiffs
        'fast_disk': [],  # used to store temporary binary file, defaults to save_path0
        'delete_bin': False,  # whether to delete binary file after processing
        'mesoscan': False,  # for reading in scanimage mesoscope files
        'bruker': False,  # whether or not single page BRUKER tiffs!
        'bruker_bidirectional': False, # bidirectional multiplane in bruker: 0, 1, 2, 2, 1, 0 (True) vs 0, 1, 2, 0, 1, 2 (False)
        'h5py': [],  # take h5py as input (deactivates data_path)
        'h5py_key': 'data',  #key in h5py where data array is stored
        'nwb_file': '', # take nwb file as input (deactivates data_path)
        'nwb_driver': '', # driver for nwb file (nothing if file is local)
        'nwb_series': '', # TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file
        'save_path0': [],  # stores results, defaults to first item in data_path
        'save_path': [],
        'save_folder': [], # directory you'd like suite2p results to be saved to
        'subfolders': [], # subfolders you'd like to search through when look_one_level_down is set to True
        'move_bin': False,  # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk

        # main settings
        'nplanes' : 1,  # each tiff has these many planes in sequence
        'nchannels' : 1,  # each tiff has these many channels per plane
        'functional_chan' : 1,  # this channel is used to extract functional ROIs (1-based)
        'tau':  1.7,  # this is the main parameter for deconvolution
        'fs': 3.51,  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        'force_sktiff': False, # whether or not to use scikit-image for tiff reading
        'frames_include': -1,
        'multiplane_parallel': False, # whether or not to run on server
        'ignore_flyback': -1,

        # output settings
        'preclassify': 0.0,  # apply classifier before signal extraction with probability 0.3
        'save_mat': 1,  # whether to save output as matlab files
        'save_NWB': False,  # whether to save output as NWB file
        'combined': True,  # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)

        # bidirectional phase offset
        'do_bidiphase': 0, #whether or not to compute bidirectional phase offset (applies to 2P recordings only)
        'bidiphase': 0, # Bidirectional Phase offset from line scanning (set by user). Applied to all frames in recording.
        'bidi_corrected': 0, # Whether to do bidirectional correction during registration

        # registration settings
        'do_registration': True,  # whether to register data (2 forces re-registration)
        'two_step_registration': 1, # whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True.
        'keep_movie_raw': 1, # whether to keep binary file of non-registered frames.
        'nimg_init': [],  # subsampled frames for finding reference image
        'batch_size': [],  # number of frames per batch
        'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1,  # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False,  # whether to save registered tiffs
        'reg_tif_chan2': False,  # whether to save channel 2 registered tiffs
        'subpixel' : 10,  # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma_time': 2,  # gaussian smoothing in time
        'smooth_sigma': 2,  # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
        'th_badframes': 1.0,  # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'norm_frames': True, # normalize frames when detecting shifts
        'force_refImg': False, # if True, use refImg stored in ops if available
        'pad_fft': False, # if True, pads image during FFT part of registration
        
        # non rigid registration settings
        'nonrigid': False,  # whether to use nonrigid registration
        'block_size': [128, 128],  # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2,  # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5,  # maximum pixel shift allowed for nonrigid, relative to rigid

        # 1P settings
        '1Preg': False,  # whether to perform high-pass filtering and tapering
        'spatial_hp_reg': 42,  # window for spatial high-pass filtering before registration
        'pre_smooth': 1,  # whether to smooth before high-pass filtering before registration
        'spatial_taper': 40,  # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])

        "diameter": 0,  # use diameter for cellpose, if 0 estimate diameter
    }