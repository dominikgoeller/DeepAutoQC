from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import nibabel as nib
import torchio as tio


class BaseBrain(ABC):
    """Generate "new" human brain as a base class for our data augmentation."""

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        self.t1w = transform_data().get_base_transform()(t1w)
        self.mask = transform_data().get_base_transform()(mask)

    # @abstractmethod
    # def compose():
    #    pass
    @abstractmethod
    def apply():
        pass


class badBrain(BaseBrain):
    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        super().__init__(t1w, mask)
        self.t1w = t1w  # we want to have the t1w with the applied changes from super()
        self.mask = (
            mask  # we want to have the mask with the applied changes from super()
        )

    # def compose(self):
    #    """Here we want to apply bad transformations to our t1w or mask or both
    #    we differentiate between scanner artefacts and articial artefacts that only transform to our mask aka red outline
    #    We want to have many different composed transforms available so we can randomly choose one of them with tio.OneOf() for our data augmentation inside the
    #    dataset for the training process.
    #    """
    #    return transform_data().choose_bad_transform()
    def apply(self):
        transf = transform_data().choose_bad_transform()
        t1w = transf(self.t1w)
        mask = transf(self.mask)
        return t1w, mask


class goodBrain(BaseBrain):
    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        super().__init__(t1w, mask)

    def compose(self):
        """See above for documentation at compose_bad()"""
        pass


@dataclass
class transform_data:
    motion: dict = field(
        default_factory=lambda: {
            "degrees": 20,
            "translation": 20,
            "num_transforms": 3,
        }
    )
    ghosting: dict = field(
        default_factory=lambda: {
            "num_ghosts": (4, 10),
            "axes": ("AP", "lr"),
            "intensity": (0.1, 0.3),
            "restore": 0.05,
        }
    )
    spike: dict = field(
        default_factory=lambda: {
            "num_spikes": 3,
            "intensity": (1, 3),
        }
    )
    affine: dict = field(
        default_factory=lambda: {
            "degrees": 20,
            "center": "image",
        }
    )
    rescale: dict = field(default_factory=lambda: {"out_min_max": (0, 1)})
    elastic: dict = field(
        default_factory=lambda: {
            "num_control_points": 7,
            "max_displacement": (7, 9, 5),
            "locked_borders": 2,
        }
    )
    swap: dict = field(
        default_factory=lambda: {
            "patch_size": 15,
            "num_iterations": 40,
        }
    )

    def get_motion(self):
        return tio.RandomMotion(
            degrees=self.motion.get("degrees"),
            translation=self.motion.get("translation"),
            num_transforms=self.motion.get("num_transforms"),
        )

    def get_ghosting(self):
        return tio.RandomGhosting(
            num_ghosts=self.ghosting.get("num_ghosts"),
            axes=self.ghosting.get("axes"),
            intensity=self.ghosting.get("intensity"),
            restore=self.ghosting.get("restore"),
        )

    def get_spike(self):
        return tio.RandomSpike(
            num_spikes=self.spike.get("num_spikes"),
            intensity=self.spike.get("intensity"),
        )

    def get_affine(self):
        return tio.RandomAffine(
            degrees=self.affine.get("degrees"),
            center=self.affine.get("center"),
        )

    def get_rescale(self):
        return tio.RescaleIntensity(
            out_min_max=self.rescale.get("out_min_max"),
        )

    def get_elastic(self):
        return tio.RandomElasticDeformation(
            num_control_points=self.elastic.get("num_control_points"),
            max_displacement=self.elastic.get("max_displacement"),
            locked_borders=self.elastic.get("locked_borders"),
        )

    def get_swap(self):
        return tio.RandomSwap(
            patch_size=self.swap.get("patch_size"),
            num_iterations=self.swap.get("num_iterations"),
        )

    def get_base_transform(self):
        return tio.Compose(
            [
                tio.ToCanonical(),
                self.get_elastic(),
            ]
        )

    def bad_scanner_dict(self):
        """Dictionary with instances of
        :class:`~torchio.transforms.Transform` as keys and
        probabilities as values. Probabilities are normalized so they sum
        to one. If a sequence is given, the same probability will be
        assigned to each transform."""
        bad_scanner_dict: dict = {
            tio.Compose(
                [
                    self.get_motion(),
                    self.get_ghosting(),
                ]
            ),
            tio.Compose(
                [
                    self.get_motion(),
                    self.get_ghosting(),
                    self.get_spike(),
                ]
            ),
            tio.Compose(
                [
                    self.get_motion(),
                    self.get_ghosting(),
                    self.get_spike(),
                    self.get_swap(),
                ]
            ),
        }
        return bad_scanner_dict

    def bad_synthetic_dict(self):
        """Dictionary with instances of
        :class:`~torchio.transforms.Transform` as keys and
        probabilities as values. Probabilities are normalized so they sum
        to one. If a sequence is given, the same probability will be
        assigned to each transform."""
        bad_synthetic_dict: dict = {
            tio.Compose([self.get_affine()]),
        }
        return bad_synthetic_dict

    def choose_bad_transform(self):
        bad_transform = tio.Compose(
            [
                tio.OneOf(
                    tio.OneOf(self.bad_scanner_dict()),
                    tio.OneOf(self.bad_synthetic_dict()),
                ),  # PROBLEM! we want to apply synthetic transforms only on mask and not t1w which is currently not the case!
                self.get_rescale(),
            ]
        )
        return bad_transform


# def motion(cfg=trf_cfg()):
#    return tio.RandomMotion(
#        degrees=cfg.motion.get("degrees"),
#        translation=cfg.motion.get("translation"),
#        num_transforms=cfg.motion.get("num_transforms"),
#    )
#
#
# def ghosting(cfg=trf_cfg()):
#    return tio.RandomGhosting(
#        num_ghosts=cfg.ghosting.get("num_ghosts"),
#        axes=cfg.ghosting.get("axes"),
#        intensity=cfg.ghosting.get("intensity"),
#        restore=cfg.ghosting.get("restore"),
#    )
#
#
# def spike(cfg=trf_cfg()):
#    return tio.RandomSpike(
#        num_spikes=cfg.spike.get("num_spikes"), intensity=cfg.spike.get("intensity")
#    )


# def affine(cfg=trf_cfg()):
#    return tio.RandomAffine(
#        degrees=cfg.affine.get("degrees"),
#        center=cfg.affine.get("center"),
#    )
#
#
# def rescale(cfg=trf_cfg()):
#    return tio.RescaleIntensity(
#        out_min_max=cfg.rescale.get("out_min_max"),
#    )
#
# def elastic(cfg=trf_cfg()):
#    return tio.RandomElasticDeformation(
#        num_control_points=cfg.elastic.get("num_control_points"),
#        max_displacement=cfg.elastic.get("max_displacement"),
#        locked_borders=cfg.elastic.get("locked_borders"),
#    )

"""Demonstration of overriding dicts from our dataclass"""
x = transform_data(
    motion={"degrees": 50, "translation": 50, "num_transforms": 7},
)  #
y = x.get_affine()
