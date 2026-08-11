# Scratch design document for YeastTiles project

## Project overview

This project provides a self-contained implementation for processing images of
yeast cells: brightfield + DAPI-stained. The goal is to build an intuitive 
pipeline for processing raw data (3D, 2 or more channels): reducing them into
much smaller 2D images. Individual yeast cells are segmented, cropped, and 
masked; this is the primary input for a neural-network based classifier to 
score each yeast tile.

## Project dependencies

This project depends on cellpose-SAM (the default 'cellpose' install from PyPI
as of August, 2026), PyQt, pyvistra, resolvde (a local project) and pytorch. 
Cellpose is used for yeast segmentation. pyvistra is used to render images.
resolvde is used to optionally deconvolve the fluorescence images.


## General pipeline

1) Raw data (3D, 2 or more channel). Default format is imaris, but pyvistra
handles the io via `pyvistra.io.load_image`. One of the channels must be a
brightfield channel because it will be used for yeast segmentation.

2a) First a focal slice would need to be extracted from the brightfield channel
to obtain the 'best' image for yeast segmentation. Because of the large field
of view typically collected for the experiment, the sample is not perfectly flat
and needs to be corrected. This is done via polynomial fitting under the folder
notesbooks/01_flatten_field.ipynb. Once corrected, a uniform focal plane with
the best defocus value for a good contrast of yeast cells is chosen (typically 
+1 micron defocus, giving dark inside and light boundary). This is the input
for cellpose for individual yeast cell segmentation.

2b) Separately, a fluorescence channel (or whatever target channel) besides
the brightfield one is then sum-projected to reduce 3D to 2D image to dramatically
reduce the size of the image. The image is then deconvolved with an equivalent
2D PSF that is also sum-projected from a 3D PSF using matching optical parameters
as the acquired data (NA, wavelength, lateral pixel spacing). The axial pixel
spacing is oversampled to ensure complete depth of focus coverage for the data.
So, if data is acquired over a 10 micron axial range, then the psf should at 
least cover 10 micron with spacing at least 2.3x Nyquist in z. This deconvolved 
image (or if data is not deconvolved) is then used with the output from 2a to 
create a yeast 'tile'

Implementation note: 2a and 2b's outputs are saved together as a single
2-channel tiff per FOV (channel 0 = flattened brightfield from 2a, channel
1 = sum-projected target from 2b) in one output folder, rather than as two
separate files in two parallel folders matched by filename stem. This
keeps the output file structure to one file per FOV and one place for the
two channels to go out of sync. Segmentation (step 3) only ever reads
channel 0; a channel-1 deconvolution step, if/when added, would apply to
that channel only and re-save the same combined file. The Cellpose GUI's
own channel picker handles opening these multi-channel files directly for
mask correction, so nothing extra is needed there either.

3) A yeast 'tile' is a homogeneously sized (64x64 pixels) from applying the
yeast segmentation results from 2a to crop a 64x64 region around each individual
cell. The mask file is a binary 'image' containing 255 pixel value for a cell
and 0 outside. The image themselves are not masked.


4) The yeast tiles are then fed into a neural network using self-supervised 
VICReg scheme to generate high-quality embeddings for sample efficient image
classification in the next step. A text file keep track of the classification of
 each tile, which is used for training a lightweight neural network doing the
 actual classification.


 ## Desired features for the project


 1) For every step 1-2a/2b, it's helpful to have a visual check before moving
 to the next stage, per image. For step 1, we need a decent multi-channel visualizer
 which can be 'borrowed' from pyvistra. For step 2, we need a way to visualize 
 and diagnose the field flattening process and tune its parameters.

 2) The corpus of data (yeast tiles) should be allowed to accumulate to gradually
 improve the classifier as more data is collected and annotated. This places
 some burden on the organization of the data. A good UI must be designed to allow
 users to "pool" tiles from various experiments (preferably organized in individual 
 folders) for a 'global' training run of the network. The networks (the embedding
 generator and the classifier) should be able to be trained separately. This is
 because we want to separate the quality of the embedding from the self-supervised
 training from the supervised training aspect for maximum flexibility.

 3) The same folder 'pooling' can be used for training cellpose models to improve
 its performance tailored to our specific experiment. However, the cellpose UI 
 itself must be used for streamlined mask correction (it generates .npy files
 for each corrected segmentation results in the same folder). Re-training the 
 cellpose network is typically done per-folder, but we want to get around that
 because we want to be able to train cellpose with a more diverse dataset that
 may be organized in different folders. This project should have a UI to easily
 train cellpose models and quickly assess its performance. Pyvistra can be used
 for the visualization here.

 4) Once cropped, we already have a decent yeast tile viewer. We'll just have to
 polish it with sensible defaults for a typical dataset acquired in our experiments.

