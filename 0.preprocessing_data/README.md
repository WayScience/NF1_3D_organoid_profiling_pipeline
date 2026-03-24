# Imaging settings

## Objective:

The olympus uplansapo 60x oil objective provides high resolution with a working distance of 0.15 M.M. and a 1.35 N.A.

## Oil immersion: 1.518

# Channel information

| Channel name   | fluorophore                    | excitation (nm) | emission (nm) | dichroic (nm) | organelle                        |
| -------------- | ------------------------------ | --------------- | ------------- | ------------- | -------------------------------- |
| Hoechst        | hoechst 33342                  | 361             | 486           | 405           | nucleus                          |
| Concanavalin A | concanavalin A alexa fluor 488 | 495             | 519           | 488           | endoplasmic reticulum            |
| WGA            | WGA alexa fluor 555            | 555             | 580           | 555           | golgi apparatus, plasma membrane |
| Phalloidin     | phalloidin alexa fluor 568     | 578             | 600           | 555           | F-actin                          |
| Mitotracker    | mitotracker deep red           | 644             | 665           | 640           | mitochondria                     |

## Deconvolution settings

The deconvolution files used can be found in the `./huygens_workflow_files` folder.

The settings in the huygens software were as follows:
| Parameter | value |
|-----------|-------|
| Algorithm | classic maximum likelihood estimation (CMLE) |
| PSF mode | theoretical |
| Max. iterations | 30 |
| Iteration mode | optimized |
| Quality change threshold | 0.01 |
| Signal to noise ratio | 26 |
| Anisotropy mode | off |
| Acuity mode | on |
| Background mode | lowest value |
| Background estimation radius | 0.7 |
| Relative background | 0.0 |
| Bleaching correction | off |
| Brick mode | auto |
| Psfs per brick mode | off |
| Psfs per brick | 1 |
| Array detector reconstruction mode | auto |

## Correct nyquist sampling

Adapted from https://svi.nl/nyquistrate

Where $n$ is the refractive index of the medium between the objective and the sample, $\alpha$ is the half-angle of the maximum cone of light that can enter or exit the objective, and $\lambda_{ex}$ is the excitation wavelength.

$$\Alpha=arcsin(NA/n)$$
$$ F*{nyquist, x,y} = \frac{1}{2 \* \delta x,y} $$
$$\Delta x,y = \frac{\lambda*{ex}}{8n \sin(\alpha)}$$
$$ F*{nyquist, x,y} = \frac{4n \sin(\alpha)}{\lambda*{ex}}$$
$$ F*{nyquist, z} = \frac{1}{2 \* \delta z} $$
$$\Delta z = \frac{\lambda*{ex}}{4n(1 - \cos(\alpha))}$$
$$ F*{nyquist, z} = \frac{2n(1 - \cos(\alpha))}{\lambda*{ex}}$$
$$N=1.518$$
$$NA=1.35$$

$$\Alpha=arcsin(1.35/1.518)=63.3 \space degrees$$
$$\Alpha=1.105 \space radians$$
| Channel name | excitation ($nm$) | $\delta x,y$ ($\mu m$) | $\delta z$ ($\mu m$)
|--------------|-----------------|-----------------------|---------------------|
| Hoechst | 361 | 0.099 | 0.299 |
| Concanavalin A | 495 | 0.121 | 0.366 |
| WGA | 555 | 0.136 | 0.411 |
| Phalloidin | 578 | 0.141 | 0.426 |
| Mitotracker | 644 | 0.157 | 0.474 |

## Run order

Each of the scripts/notebooks in this module are run in the following order:

- 0.Patient_specific_preprocessing.py
- 1.Make_zstack_and_copy_over.py
- 1Z.make_zstack_and_copy_over_CQ1.py (if using CQ1 data)
- 2.Perform_file_corruption_checks.py
- 3.Decon_preprocessing.py
- Here is when I run the huygens deconvolution software in batch mode
- 4.Decon_post_processing.py

Scripts in `./scripts` are generated from the notebooks in `./notebooks` via `jupyter nbconvert` and may not be present until preprocessing is run.

Please see the `nyquist_sampling_calculations.ipynb` notebook for the calculations of the nyquist sampling rates.

## Well fovs that were removed due to errors during preprocessing

| Patient     | Well_FOV | status  | reason         |
| ----------- | -------- | ------- | -------------- |
| NF0014_T1   | F11-3    | removed | unnest failure |
| NF0016_T1   | D3-2     | removed | unnest failure |
| NF0016_T1   | G3-2     | removed | unnest failure |
| NF0016_T1   | C5-3     | removed | unnest failure |
| NF0016_T1   | D8-1     | removed | unnest failure |
| NF0018_T6   | F4-2     | removed | unnest failure |
| NF0018_T6   | C8-2     | removed | unnest failure |
| NF0021_T1   | E10-2    | removed | unnest failure |
| NF0021_T1   | C3-4     | removed | unnest failure |
| NF0030_T1   | C6-4     | removed | unnest failure |
| NF0030_T1   | D4-7     | removed | unnest failure |
| NF0035_T1   | E9-3     | removed | unnest failure |
| SARCO361_T1 | D2-3     | removed | unnest failure |

## Yokogawa CQ1 vs echo analysis and key differences

| Feature                     | yokogawa CQ1                | echo                   |
| --------------------------- | --------------------------- | ---------------------- |
| Imaging modality            | spinning disk confocal      | spinning disk confocal |
| Objective                   | 60x oil immersion (1.35 NA) | 60x air (0.9 NA)       |
| Medium and refractive index | oil (1.518)                 | air (1.0)              |
| Pixel size                  | 0.1006 μm                   | 0.1083 μm              |
| Z-step size                 | 1 μm                        | 1 μm                   |

Please not that water's RI is approximately 1.33 and matrigel's RI is approximately 1.34-1.36.
