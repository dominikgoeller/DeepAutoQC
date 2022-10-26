import matplotlib as mpl
import nibabel as nib
import nilearn._utils.niimg as niimg
import nilearn.plotting.displays._slicers as slicers
import numpy as np
from nilearn.plotting import plot_anat, view_img
from niworkflows.utils.images import rotate_affine, rotation2canonical
from niworkflows.viz.utils import cuts_from_bbox, robust_set_limits
from seaborn import color_palette


# performance 1
def _is_binary_niimg(img):
    return img.header.get_data_dtype().kind != "f"


niimg._is_binary_niimg = _is_binary_niimg
slicers._is_binary_niimg = _is_binary_niimg

# performance 2
mpl.use("agg")  # non interactive backend only to save files
# performance 3


def _robust_set_limits(array, plot_params):
    vmin, vmax = np.percentile(array, (15, 99.8))
    plot_params["vmin"] = vmin
    plot_params["vmax"] = vmax

    return plot_params


def _plot_anat_with_contours(image, segs=None, **plot_params):
    nsegs = len(segs or [])
    plot_params = plot_params or {}
    # plot_params' values can be None, however they MUST NOT
    # be None for colors and levels from this point on.
    colors = plot_params.pop("colors", None) or []
    levels = plot_params.pop("levels", None) or []
    missing = nsegs - len(colors)
    if missing > 0:  # missing may be negative
        colors = colors + color_palette("husl", missing)

    colors = [[c] if not isinstance(c, list) else c for c in colors]

    if not levels:
        levels = [[0.5]] * nsegs

    # anatomical
    display = plot_anat(image, **plot_params)

    # remove plot_anat -specific parameters
    plot_params.pop("display_mode")
    plot_params.pop("cut_coords")

    plot_params["linewidths"] = 0.5
    for i in reversed(range(nsegs)):
        plot_params["colors"] = colors[i]
        display.add_contours(segs[i], levels=levels[i], **plot_params)

    return display


target_width = 2047  # change to 2048 results in distorted x achsis because of image.resize() probably


def to_rgb(display):
    figure = display.frame_axes.figure
    canvas = figure.canvas

    # scale to target_width
    width, height = canvas.get_width_height()
    figure.set_dpi(target_width / width * figure.get_dpi())

    canvas.draw()
    width, height = canvas.get_width_height()

    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
        (height, width, -1)
    )[..., :3]

    image = image[:, :target_width, :]  # crop rounding errors
    # image = image.copy()
    # image.resize(height, target_width, 3)
    return image


# since we nib.load(t1wpath) already in our dataset we can redefine this function (substitute t1w_path for t1w)
def to_image(t1w, mask_path):
    plot_params = dict(colors=None)

    # image_nii: nib.Nifti1Image = nib.load(t1w_path)
    image_nii = t1w
    seg_nii: nib.Nifti1Image = nib.load(mask_path)

    canonical_r = rotation2canonical(image_nii)
    image_nii = rotate_affine(image_nii, rot=canonical_r)
    seg_nii = rotate_affine(seg_nii, rot=canonical_r)

    data = image_nii.get_fdata()
    plot_params = _robust_set_limits(data, plot_params)

    bbox_nii = seg_nii

    cuts = cuts_from_bbox(bbox_nii, cuts=7)

    images = list()
    for d in plot_params.pop("dimensions", ("z", "x", "y")):
        plot_params["display_mode"] = d
        plot_params["cut_coords"] = cuts[d]
        display = _plot_anat_with_contours(image_nii, segs=[seg_nii], **plot_params)
        images.append(to_rgb(display))
        display.close()

    image = np.vstack(images)
    return image
