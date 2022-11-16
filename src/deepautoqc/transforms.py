import nibabel as nib
import torchio as tio


class BaseBrain:
    """Base class for initial transforms to generate "new" human brain"""

    trf_C = tio.Compose(
        [
            tio.ToCanonical(),
            tio.RandomElasticDeformation(
                num_control_points=(7), max_displacement=(7, 9, 5), locked_borders=2
            ),
        ]
    )

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        self.t1w = self.trf_C(t1w)  # use nib.load()
        self.mask = self.trf_C(
            mask
        )  # use nib.load() and change nib.load(mask) in to_image!!


class SubBrain(BaseBrain):
    # transforms dict?
    def __init__(self, *args, **kwargs):
        super(SubBrain, self).__init__(*args, **kwargs)
        self.sub_t1w = None  # self.t1w
        self.sub_mask = None  # self.mask


class transforms_cfg:
    t1_real = tio.Compose(
        [
            tio.RandomMotion(degrees=20, translation=20, num_transforms=3),
            tio.RandomGhosting(
                num_ghosts=(4, 10),
                axes=("AP", "lr"),
                intensity=(0.1, 0.3),
                restore=0.05,
            ),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ]
    )
    t2_real = tio.Compose(
        [
            tio.RandomMotion(degrees=20, translation=20, num_transforms=3),
            tio.RandomGhosting(
                num_ghosts=(4, 10),
                axes=("AP", "lr"),
                intensity=(0.1, 0.3),
                restore=0.05,
            ),
            tio.RandomSpike(num_spikes=(3), intensity=(1, 3)),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ]
    )

    real_artifacts = {
        t1_real,
        t2_real,
    }  # for further use in data.py tio.OneOf(real_artifacts)

    # artificial artifacts for red outline
    t1_ai = tio.Compose(
        [
            tio.RandomAffine(degrees=(20), center="image"),
            tio.RescaleIntensity(out_min_max=(0, 1)),
        ]
    )

    # wait with further transforms until meeting
    ai_artifacts = {t1_ai}


## functions nach diesem Muster definieren?
trf_cfg = {
    "motion": {
        "degrees": 20,
        "translation": 20,
        "num_transforms": 3,
    },
    "ghosting": {},
}


def motion(cfg: dict = trf_cfg):
    return tio.RandomMotion(
        degrees=cfg["motion"]["degrees"],
        translation=cfg["motion"]["translation"],
        num_transforms=cfg["motion"]["num_transforms"],
    )
