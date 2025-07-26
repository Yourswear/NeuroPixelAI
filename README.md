# NeuroPixelAI
[[PyPI version]](https://pypi.org/project/neuroai/)

NeuroPixelAI is an open-source, GPU-accelerated pipeline that converts raw mesoscale imaging data into spike trains via automated motion registration, 
deep-learning denoising, region-adaptive segmentation, baseline-corrected trace extraction, and spike inference.

This code was written by Jiahao Hu from Li lab at Fudan University.All related implementations are referenced in the paper.

The reference paper is [here]([https://www.biorxiv.org/content/early/2017/07/20/061507](https://www.biorxiv.org/content/10.1101/2025.07.03.662899v1)).

## CITATION
If you use this package in your research, please cite the [paper]([https://www.biorxiv.org/content/early/2017/07/20/061507](https://www.biorxiv.org/content/10.1101/2025.07.03.662899v1)):

Jiahao Hu, Yanfeng Zhu, Shoupei Liu, Chengyu Li, Min Zhang, Xinyang Gu, Jingchuan Wu, Fang Xu, Ying Mao, Bo Li (2025) Cortex-Wide, Cellular-Resolution Volumetric Imaging with a Modular Two-Photon Imaging Platform
BioRxiv 662899

## Local Installation
1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.8** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Create a new environment with `conda create --name neuroai python=3.8`.
4. To activate this new environment, run `conda activate neuronai`
5. Install the NeuroPixelAI with `python -m pip install neuroai`.
6. Next, install the appropriate version of PyTorch and CUDA support based on your GPU configuration. For example, to install PyTorch 2.7.1 with CUDA 11.8, run the following command: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`.
7. Now run `python -m NeuroPixelAI` and you're all set.

Notably, to avoid potential errors caused by hardware incompatibility, it is strongly recommended to run NeuroPixelAI on Windows systems with NVIDIA GPUs.

## Examples

An example dataset is provided [here](). It's a single-plane, single-channel recording.

## Getting started

After completing the installation steps, follow these instructions to launch NeuroPixelAI:

1.  **Activate the environment:** Open Anaconda Prompt and enter:
    ```
    activate neuroai
    ```
2.  **Launch NeuroPixelAI:** Run:
    ```
    python -m NeuroPixelAI
    ```
A standard calcium imaging data processing workflow consists of five stages: **motion correction**, **image denoising**, **cell segmentation**, **signal extraction**, and **spike inference**. For a basic usage example, refer to [Supplementary Viedo 3](https://www.neuronai.com).

### Tab 1: Data Loading, Motion Correction & Denoising
1.  **Load Data:** Click `Load Image` in the bottom-left corner to select your data file. The raw image will appear in the left image panel.
2.  **Run Processing:** Adjust parameters in the left-side `Parameter` panel, then click `Run Enhancement`. This performs motion correction and denoising. The processed image will display in the right image panel.

### Tab 2: Segmentation, Extraction & Spike Analysis
1.  **Run Segmentation:** After adjusting parameters, click `Run segmentation`.
2.  **Extract Signals:** Click `Signal extraction`.
3.  **Infer Spikes:** Click `Spike inference`.
4.  **Discretize Spikes:** Click `Spike discrete`.

### Additional Tab 2 Controls
*   **Manual ROI Selection:** Right-click within the top-left image panel to manually select Regions of Interest (ROIs).
*   **Adjust ROIs:** Use the `W`, `A`, `S`, `D` keys to move individual or all ROIs. Press `E` to delete a single ROI.
    *   *Note:* If interactions become unresponsive, simply click any checkbox within Tab 2 twice to restore functionality.
*   **Select ROIs to View:** Enter specific ROI identifiers (e.g., `'a,b,e:h'`) in the `ROI to Show` field to view corresponding signals.

All processed data can be exported to `.mat` files and previously saved `.mat` files can be reloaded into NeuroPixelAI using the `Load processed result` option.You can also modify the exported `.mat` files using external tools and then re-import them into NeuroPixelAI for further analysis or visualization.

## Parameter Specifications

This section details the key parameters for each processing stage. Parameters marked with 🔴 are commonly adjusted for optimal results.

### Motion Correction Parameters

*   **`Smooth_sigma_time`** (Default: `2`)
    *   Gaussian **temporal** standard deviation. Smoothes the data **over time** before calculating phase correlation.
    *   *Recommendation:* Use `1` or `2` for low signal-to-noise ratio (SNR) data.
*   🔴 **`Nimg_init`** (Default: `500`)
    *   Number of frames used to compute the **initial reference image**. Registration works by aligning each frame to this reference.
*   **`Maxregshift`** (Default: `0.1`)
    *   Maximum **relative** shift allowed (as a fraction of the image dimension).
    *   *Example:* With a default of `0.1` and data size `512x512`, the maximum shift per frame is `51.2` pixels.
*   **`Smooth_sigma`** (Default: `2`)
    *   Gaussian **spatial** standard deviation. Smoothes the phase correlation map between the reference and each frame.
*   ☑️ **`Nonrigid`** (Checkbox)
    *   Enable **non-rigid registration** to correct for small sample deformations during imaging.
*   ☑️ **`Bidiphase`** (Checkbox)
    *   Correct for **row misalignment artifacts** (scan phase offset between odd and even lines).

### Denoising Parameters

*   **`Denoise model`** (Default: `deepcad_model`)
    *   Specifies the denoising model used by **DeepCAD-RT**.

### Cell Segmentation Parameters

*   **`Flow_threshold`** (Default: `0.4`)
    *   Maximum allowed **flow error** per potential object during segmentation.
    *   *Effect:* Increasing this value → **More ROIs** detected. Decreasing this value → **Fewer ROIs** detected.
*   🔴 **`Cellprob_threshold`** (Default: `0`)
    *   Threshold (range: `-6` to `6`) for the probability that an object is a valid cell.
    *   *Adjustment:*
        *   **Decrease** if too few ROIs are found.
        *   **Increase** if too many ROIs are found, especially in dark areas.
*   🔴 **`Diameter`** (Default: `21.5`)
    *   Estimated **diameter** of objects (e.g., cell bodies) in **pixels**.
*   **`Segmentation model`** (Default: `cyto`)
    *   Specifies the **Cellpose 2.0** model used (`cyto` is the default model for neuronal cell bodies).
*   ☑️ **`Regional Seg`** (Checkbox)
    *   Enable **regional segmentation** mode. Recommended for processing **large field-of-view (FOV)** data (e.g., `2048x2048` or `4096x4096` pixels).

### Spike Inference Parameters

*   🔴 **`Spike model`** (Default: `3.51Hz_420ms`)
    *   Specifies the model used by **Cascade** for spike inference.
    *   *Note:* Models near `FrameRate (Hz) x SpikeDecayTime (ms) ≈ 1500` (e.g., `3.51 x 420 ≈ 1500`) often perform best.

### Additional Options

*   ☑️ **`Baseline correction`** (Checkbox)
    *   Enable **baseline correction** during signal extraction. Recommended for **long recordings**.
*   ☑️ **`Mask fixed`** (Checkbox)
    *   Use a **fixed segmentation mask** when processing multiple files in a batch (`Run batches` mode).
