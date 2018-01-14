"""Script to create a normalised dataset

The normalisation in this script follows the guideline for
histology images normalisation from the following article:
Macenko M, Niethammer M, Marron JS, Borland D, Woosley JT, Guan X, et al. A method for normalizing
histology slides for quantitative analysis. In: Proceedings—2009 IEEE International Symposium on
Biomedical Imaging: From Nano to Macro, ISBI 2009. Boston, Massachusetts; 2009. p. 1107–1110.
"""

import numpy as np
import os
from scipy import misc
import scipy
from sklearn.decomposition import PCA
from typing import TypeVar

PathLike = TypeVar('PathLike', str, bytes)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, os.path.pardir, "dataset")
DATA_SUB_DIRS = ["benign", "insitu", "invasive", "normal"][1:]
assert os.path.exists(DATA_DIR)
NORM_DATA_DIR = os.path.join(CURRENT_DIR, os.path.pardir, "norm_dataset")
if not os.path.exists(NORM_DATA_DIR):
    os.makedirs(NORM_DATA_DIR)


def normalize_staining(sample_tuple, beta=0.15, alpha=1, light_intensity=255):
    """Normalize the staining of H&E histology slides.

    This function normalizes the staining of H&E histology slides.
    Code from: https://github.com/SparkTC/deep-histopath/blob/master/breastcancer/preprocessing.py

    References
    ----------

      - Macenko, Marc, et al. "A method for normalizing histology slides
      for quantitative analysis." Biomedical Imaging: From Nano to Macro,
      2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
        - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
      - https://github.com/mitkovetta/staining-normalization

    Parameters
    ----------
      sample_tuple:
        A (slide_num, sample) tuple, where slide_num is an
        integer, and sample is a 3D NumPy array of shape (H,W,C).

    Returns
    -------
      A (slide_num, sample) tuple, where the sample is a 3D NumPy array
      of shape (H,W,C) that has been stain normalized.
    """
    # Setup.
    slide_num, sample, name = sample_tuple
    x = np.asarray(sample)
    # print(x)

    h, w, c = x.shape
    x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)


    # Reference stain vectors and stain saturations.  We will normalize all slides
    # to these references.  To create these, grab the stain vectors and stain
    # saturations from a desirable slide.

    # Values in reference implementation for use with eigendecomposition approach, natural log,
    # and `light_intensity=240`.
    # stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
    # max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

    # SVD w/ log10, and `light_intensity=255`.
    stain_ref = (np.array([0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629])
                 .reshape(3, 2))
    max_sat_ref = np.array([0.82791151, 0.61137274]).reshape(2, 1)

    # Convert RGB to OD.
    # Note: The original paper used log10, and the reference implementation used the natural log.
    # OD = -np.log((x+1)/light_intensity)  # shape (H*W, C)
    OD = -np.log10(x / light_intensity + 1e-8)
    # print(OD)


    # Remove data with OD intensity less than beta.
    # I.e. remove transparent pixels.
    # Note: This needs to be checked per channel, rather than
    # taking an average over all channels for a given pixel.
    # Use a loop to check to have enough elements.
    while True:
        OD_thresh = OD[np.all(OD >= beta, 1), :]  # shape (K, C)
        if len(OD_thresh) / len(OD) < 0.001:
            print("\t", name, len(OD_thresh) / len(OD), "- beta:", beta)
            beta -= 0.005
        else:
            break

    # Calculate eigenvectors.
    # Note: We can either use eigenvector decomposition, or SVD.
    # eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
    U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

    # Extract two largest eigenvectors.
    # Note: We swap the sign of the eigvecs here to be consistent
    # with other implementations.  Both +/- eigvecs are valid, with
    # the same eigenvalue, so this is okay.
    # top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
    top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

    # Project thresholded optical density values onto plane spanned by
    # 2 largest eigenvectors.
    proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

    # Calculate angle of each point wrt the first plane direction.
    # Note: the parameters are `np.arctan2(y, x)`
    angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

    # Find robust extremes (a and 100-a percentiles) of the angle.
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    # Convert min/max vectors (extremes) back to optimal stains in OD space.
    # This computes a set of axes for each angle onto which we can project
    # the top eigenvectors.  This assumes that the projected values have
    # been normalized to unit length.
    extreme_angles = np.array(
        [[np.cos(min_angle), np.cos(max_angle)],
         [np.sin(min_angle), np.sin(max_angle)]]
    )  # shape (2,2)
    stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

    # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
    if stains[0, 0] < stains[0, 1]:
        stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

    # Calculate saturations of each stain.
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where `OD` is the matrix of optical density values of our image,
    # `V` is the matrix of stain vectors, and `S` is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    sats, _, _, _ = np.linalg.lstsq(stains, OD.T, rcond=None)

    # Normalize stain saturations to have same pseudo-maximum based on
    # a reference max saturation.
    max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
    sats = sats / max_sat * max_sat_ref

    # Compute optimal OD values.
    OD_norm = np.dot(stain_ref, sats)

    # Recreate image.
    # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
    # not return the correct values due to the initital values being outside of [0,255].
    # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
    # same behavior as Matlab.
    # x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
    x_norm = 10 ** (-OD_norm) * light_intensity - 1e-8  # log10 approach
    x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
    x_norm = x_norm.astype(np.uint8)
    x_norm = x_norm.T.reshape(h, w, c)
    return (slide_num, x_norm)


def load_and_normalise(subdir: PathLike, image_name: PathLike):
    # Load image
    image_matrix = misc.imread(os.path.join(DATA_DIR, subdir, image_name))
    try:
        _, image_matrix = normalize_staining((0, image_matrix, image_name))
        # Check that dirs exist
        if not os.path.exists(os.path.join(NORM_DATA_DIR, subdir)):
            os.makedirs(os.path.join(NORM_DATA_DIR, subdir))

        # Save the normalised image
        misc.imsave(os.path.join(NORM_DATA_DIR, subdir, image_name), image_matrix)
    except Exception as e:
        print(subdir, image_name, end=" - ")
        print("Thrown exception:", repr(e))


def batch_normalise():
    print("Batch Normalization")
    print("-------------------")
    print("Base path:", DATA_DIR)
    print("Normalization path:", NORM_DATA_DIR)
    print("Found folders:", DATA_SUB_DIRS)
    for n_dir, subdir in enumerate(DATA_SUB_DIRS):
        images_list = os.listdir(os.path.join(DATA_DIR, subdir))
        for i, file_name in enumerate(images_list):
            if file_name.endswith(".jpeg"):

                load_and_normalise(subdir, file_name)

                if i != 0 and i+1 % 25 == 0:
                    percent_complete = (((i + 1) + (len(images_list) * n_dir))
                                        / (len(images_list) * len(DATA_SUB_DIRS))) * 100
                    print("{}/{}: {:.2f}% images normalised".format(i+1,
                                                                    len(images_list) * len(DATA_SUB_DIRS),
                                                                    percent_complete))


def batch_normalise_as_list(images_list):
    for i, path_tuple in enumerate(images_list):

        load_and_normalise(*path_tuple)

        if i != 0 and i % 5 == 0:
            percent_complete = ((i + 1)
                                / (len(images_list))) * 100
            print("{}/{}: {:.2f}% images normalised".format(i + 1,
                                                            len(images_list),
                                                            percent_complete))


if __name__ == "__main__":
    batch_normalise()

    # batch_normalise_as_list([("insitu", "is098.jpeg")])
    # batch_normalise_as_list([("insitu", "is095.jpeg")])
    # batch_normalise_as_list([("insitu", "is094.jpeg")])
    # batch_normalise_as_list([("insitu", "is093.jpeg")])



