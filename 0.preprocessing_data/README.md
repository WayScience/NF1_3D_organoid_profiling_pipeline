# Imaging settings

## Objective:
The Olympus UPlanSApo 60x oil objective provides high resolution with a working distance of 0.15 M.M. and a 1.35 N.A.
## Oil immersion: 1.518

# Channel information
| Channel Name | Fluorophore | Excitation (nm) | Emission (nm) | Dichroic (nm) | Organelle |
|--------------|-------------|-----------------|---------------|----------------|------------|
| Hoechst     | Hoechst 33342 | 361             | 486           | 405            | Nucleus    |
| Concanavalin A | Concanavalin A Alexa Fluor 488 | 495             | 519           | 488            | Endoplasmic Reticulum |
| WGA        | WGA Alexa Fluor 555 | 555             | 580           | 555            | Golgi Apparatus, Plasma Membrane |
| Phalloidin | Phalloidin Alexa Fluor 568 | 578             | 600           | 555            | F-actin    |
| MitoTracker | MitoTracker Deep Red  | 644             | 665           | 640            | Mitochondria |

## Deconvolution settings
The deconvolution files used can be found in the `./1.huygens_workflow_files` folder.

The settings in the Huygens software were as follows:
| Parameter | Value |
|-----------|-------|
| Algorithm | Classic Maximum Likelihood Estimation (CMLE) |
| PSF mode | Theoretical |
| Max. iterations | 30 |
| iteration mode | Optimized |
| Quality change threshold | 0.01 |
| Signal to noise ratio | 26 |
| Anisotropy mode | Off |
| Acuity mode | On |
| Background mode | Lowest value |
| Background estimation radius | 0.7 |
| Relative background | 0.0 |
| Bleaching correction | Off |
| Brick mode | Auto |
| PSFs per brick mode | Off |
| PSFs per brick | 1 |
| Array detector reconstruction mode | Auto |

## Correct Nyquist sampling
Adapted from https://svi.nl/NyquistRate

Where $n$ is the refractive index of the medium between the objective and the sample, $\alpha$ is the half-angle of the maximum cone of light that can enter or exit the objective, and $\lambda_{ex}$ is the excitation wavelength.

$$\alpha=arcsin(NA/n)$$
$$ F_{Nyquist, x,y} = \frac{1}{2 * \Delta x,y} $$
$$\Delta x,y = \frac{\lambda_{ex}}{8n \sin(\alpha)}$$
$$ F_{Nyquist, x,y} = \frac{4n \sin(\alpha)}{\lambda_{ex}}$$
$$ F_{Nyquist, z} = \frac{1}{2 * \Delta z} $$
$$\Delta z = \frac{\lambda_{ex}}{4n(1 - \cos(\alpha))}$$
$$ F_{Nyquist, z} = \frac{2n(1 - \cos(\alpha))}{\lambda_{ex}}$$
$$n=1.518$$
$$NA=1.35$$

$$\alpha=arcsin(1.35/1.518)=63.3 \space degrees$$
$$\alpha=1.105 \space radians$$
| Channel Name | Excitation ($nm$) | $\Delta x,y$ ($\mu m$) | $\Delta z$ ($\mu m$)
|--------------|-----------------|-----------------------|---------------------|
| Hoechst     | 361             | 0.099                 | 0.299               |
| Concanavalin A | 495             | 0.121                 | 0.366               |
| WGA        | 555             | 0.136               | 0.411               |
| Phalloidin | 578             | 0.141                 | 0.426               |
| MitoTracker | 644             | 0.157                 | 0.474               |

## Run order
Each of the scripts/notebooks in this module are run in the following order:
- 0.patient_specific_preprocessing.py
- 1.update_file_structure.py
- 2a.make_z-stack_images.py
- 2b.perform_file_corruption_checks.py
- 3.decon_preprocessing.py
- Here is when I run the Huygens deconvolution software in batch mode
- 4.post_decon_preprocessing.py

Please see the `nyquist_sampling_calculations.ipynb` notebook for the calculations of the Nyquist sampling rates.


## Well FOVs that were removed due to errors during preprocessing
| Patient | Well_FOV | Status | Reason |
|---------|----------|--------|--------|
| NF0014_T1 | F11-3 | removed | Unnest failure |
| NF0016_T1 | D3-2 | removed | Unnest failure |
| NF0016_T1 | G3-2 | removed | Unnest failure |
| NF0016_T1 | C5-3 | removed | Unnest failure |
| NF0016_T1 | D8-1 | removed | Unnest failure |
| NF0018_T6 | F4-2 | removed | Unnest failure |
| NF0018_T6 | C8-2 | removed | Unnest failure |
| NF0021_T1 | E10-2 | removed | Unnest failure |
| NF0021_T1 | C3-4| removed | Unnest failure |
| NF0030_T1 | C6-4 | removed | Unnest failure |
| NF0030_T1 | D4-7 | removed | Unnest failure |
| NF0035_T1 | E9-3 | removed | Unnest failure |


## Yokogawa CQ1 vs Echo analysis and key differences
| Feature | Yokogawa CQ1 | Echo |
|---------|--------------|------|
| Imaging Modality | Spinning Disk Confocal | Spinning Disk Confocal |
| Objective | 60x Oil Immersion (1.35 NA) | 60x Air (0.9 NA) |
| Medium and Refractive Index | Oil (1.518) | Air (1.0) |
| Pixel Size | 0.1006 μm | 0.1083 μm |
| Z-step Size | 1 μm | 1 μm |

Please not that water's RI is approximately 1.33 and Matrigel's RI is approximately 1.34-1.36.

