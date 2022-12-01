from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import nibabel as nib
import torchio as tio


class BaseBrain(ABC):
    """Generate "new" human brain as a base class for our data augmentation."""

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        self.t1w = transform_data().get_base_transform()(
            nib.load(t1w)
        )  # load t1w from path
        self.mask = transform_data().get_base_transform()(
            nib.load(mask)
        )  # load mask from path

    # @abstractmethod
    # def compose():
    #    pass
    @abstractmethod
    def apply():
        pass


class BadScannerBrain(BaseBrain):
    """Apply label changing transforms on both t1w and mask."""

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        super().__init__(t1w, mask)

    def apply(self):
        transf = transform_data().get_bad_scanner_transform()
        # transf = tio.OneOf(transforms=transform_data().bad_scanner_dict())
        t1w = transf(self.t1w)
        mask = self.mask
        return t1w, mask


class BadSyntheticBrain(BaseBrain):
    """Synthetic label changing transforms which are only applied to the mask"""

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        super().__init__(t1w, mask)

    def apply(self):
        transf = transform_data().get_bad_synthetic_transform()
        t1w = self.t1w
        mask = transf(self.mask)
        return t1w, mask


class GoodScannerBrain(BaseBrain):
    """Non-label changing class that mirrors the label-changing scanner transforms up to a certain degree"""

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        super().__init__(t1w, mask)

    def apply(self):
        transf = transform_data().get_good_scanner_transform()
        t1w = transf(self.t1w)
        mask = self.mask
        return t1w, mask


class GoodSyntheticBrain(BaseBrain):
    """Non-label changing class that mirrors the label-changing synthetic transforms up to a certain degree"""

    def __init__(self, t1w: nib.Nifti1Image, mask: nib.Nifti1Image):
        super().__init__(t1w, mask)

    def apply(self):
        transf = transform_data().get_good_synthetic_transform()
        t1w = self.t1w
        mask = transf(self.mask)
        return t1w, mask


@dataclass
class transform_data:
    motion: dict = field(
        default_factory=lambda: {
            "degrees": 20,
            "translation": 30,
            "num_transforms": 3,
        }
    )
    ghosting: dict = field(
        default_factory=lambda: {
            "num_ghosts": (2, 5),
            # "axes": (0,1),
            "intensity": (0.5, 1),
            "restore": 0.05,
        }
    )
    spike: dict = field(
        default_factory=lambda: {
            "num_spikes": (1, 1),
            "intensity": (0.6, 0.9),
        }
    )
    affine: dict = field(
        default_factory=lambda: {
            "degrees": 5,
            "scales": (1, 1.2),
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
            "patch_size": 10,
            "num_iterations": 40,
        }
    )
    noise: dict = field(
        default_factory=lambda: {
            "mean": 0,
            "std": (0, 15),
        }
    )
    blur: dict = field(default_factory=lambda: {"std": (0, 0.3)})

    def get_blur(self):
        return tio.RandomBlur(
            std=self.blur.get("std"),
        )

    def get_noise(self):
        return tio.RandomNoise(
            mean=self.noise.get("mean"),
            std=self.noise.get("std"),
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
            # axes=self.ghosting.get("axes"),
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
            scales=self.affine.get("scales"),
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
                tio.ToCanonical(),  # not needed??
                self.get_elastic(),
            ]
        )

    def bad_scanner_dict(self):
        """Dictionary with instances of
        :class:`~torchio.transforms.Transform` as keys and
        probabilities as values. Probabilities are normalized so they sum
        to one. If a sequence is given, the same probability will be
        assigned to each transform."""
        b_scan_dict: dict = {
            tio.Compose(
                [
                    self.get_motion(),
                    self.get_ghosting(),
                ]
            ): 1,
            tio.Compose(
                [
                    self.get_ghosting(),
                    self.get_swap(),
                ]
            ): 1,
            tio.Compose(
                [
                    self.get_spike(),
                    self.get_ghosting(),
                ]
            ): 1,
        }
        return b_scan_dict

    def bad_synthetic_dict(self):
        """Dictionary with instances of
        :class:`~torchio.transforms.Transform` as keys and
        probabilities as values. Probabilities are normalized so they sum
        to one. If a sequence is given, the same probability will be
        assigned to each transform."""
        b_syn_dict: dict = {
            self.get_affine(): 1,
            transform_data(
                affine={"degrees": 5, "scales": (1, 1.3), "center": "image"}
            ).get_affine(): 1,
            transform_data(
                affine={"degrees": 5, "scales": (0.8, 1), "center": "image"}
            ).get_affine(): 1,
            transform_data(
                affine={"degrees": 5, "scales": (0.7, 1), "center": "image"}
            ).get_affine(): 1,
        }
        return b_syn_dict

    def good_synthetic_dict(self):
        """Dictionary with instances of
        :class:`~torchio.transforms.Transform` as keys and
        probabilities as values. Probabilities are normalized so they sum
        to one. If a sequence is given, the same probability will be
        assigned to each transform."""
        g_syn_dict: dict = {
            transform_data(
                affine={"degrees": 5, "scales": (1, 1.02), "center": "image"}
            ).get_affine(): 1,
            transform_data(
                affine={"degrees": 5, "scales": (1, 1.05), "center": "image"}
            ).get_affine(): 1,
            transform_data(
                affine={"degrees": 5, "scales": (0.95, 1), "center": "image"}
            ).get_affine(): 1,
            transform_data(
                affine={"degrees": 5, "scales": (0.98, 1), "center": "image"}
            ).get_affine(): 1,
        }
        return g_syn_dict

    def good_scanner_dict(self):
        """Dictionary with instances of
        :class:`~torchio.transforms.Transform` as keys and
        probabilities as values. Probabilities are normalized so they sum
        to one. If a sequence is given, the same probability will be
        assigned to each transform."""
        g_scan_dict: dict = {
            tio.Compose(
                [
                    transform_data(
                        spike={"num_spikes": (1, 1), "intensity": (0, 0.25)}
                    ).get_spike(),
                    transform_data(
                        ghosting={
                            "num_ghosts": (1, 1),
                            "intensity": (0, 0.25),
                            "restore": 0.5,
                        }
                    ).get_ghosting(),
                    self.get_blur(),
                    self.get_noise(),
                ]
            ): 1,
            tio.Compose(
                [
                    transform_data(
                        spike={"num_spikes": (1, 1), "intensity": (0, 0.5)}
                    ).get_spike(),
                    transform_data(
                        ghosting={
                            "num_ghosts": (1, 1),
                            "intensity": (0, 0.5),
                            "restore": 0.5,
                        }
                    ).get_ghosting(),
                    transform_data(blur={"std": (0, 0.5)}).get_blur(),
                    transform_data(noise={"mean": 0, "std": (0, 0.3)}).get_noise(),
                ]
            ): 1,
            tio.Compose(
                [
                    transform_data(
                        spike={"num_spikes": (1, 1), "intensity": (0, 0.7)}
                    ).get_spike(),
                    transform_data(
                        ghosting={
                            "num_ghosts": (1, 1),
                            "intensity": (0, 0.7),
                            "restore": 0.5,
                        }
                    ).get_ghosting(),
                    transform_data(blur={"std": (0, 0.5)}).get_blur(),
                    transform_data(noise={"mean": 0, "std": (0, 0.3)}).get_noise(),
                ]
            ): 1,
        }
        return g_scan_dict

    def get_bad_scanner_transform(self):
        transf = tio.Compose(
            [
                tio.OneOf(self.bad_scanner_dict()),
                self.get_rescale(),
            ]
        )
        return transf

    def get_bad_synthetic_transform(self):
        transf = tio.Compose(
            [
                tio.OneOf(self.bad_synthetic_dict()),
                self.get_rescale(),
            ]
        )
        return transf

    def get_good_scanner_transform(self):
        transf = tio.Compose(
            [
                tio.OneOf(self.good_scanner_dict()),
                self.get_rescale(),
            ]
        )
        return transf

    def get_good_synthetic_transform(self):
        transf = tio.Compose(
            [
                tio.OneOf(self.good_synthetic_dict()),
                self.get_rescale(),
            ]
        )
        return transf
