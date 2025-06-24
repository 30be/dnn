import warnings

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as functional

from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna import elements


class DatasetOfWavefronts(Dataset):

    def __init__(
        self,
        init_ds: Dataset,
        transformations: transforms.Compose,
        sim_params: SimulationParameters,
        target: str = 'label',
        detector_mask: torch.Tensor | None = None
    ):
        """
        Parameters
        ----------
        init_ds : torch.utils.data.Dataset
            An initial dataset (of images and labels).
        transformations : transforms.Compose
            A sequence of transforms that will be applied to dataset elements (images).
        sim_params : SimulationParameters
            Simulation parameters for a further optical network.
        terget : str
            A type of target
                (1) 'label' - returns just a number of class
                (2) 'detector' - returns an expected detector picture
        detector_mask: torch.Tensor | None
            A detector mask to generate target images (if tardet == 'detector')
        """
        self.init_ds = init_ds
        self.transformations = transformations

        self.sim_params = sim_params  # to check if all transforms results in right shape
        self.check_transformations()  # print warnings if necessary

        self.target = target
        self.detector_mask = detector_mask

    def check_transformations(self):
        """
        Checks if transformations transforms an image to a right-shaped Wavefront.
        """
        random_image = functional.to_pil_image(torch.rand(size=(5, 5)))  # random image
        wavefront = self.transformations(random_image)

        # check type
        if not isinstance(wavefront, Wavefront):
            warnings.warn(
                message='An output object is not of the Wavefront type!'
            )

        # compare nodes number of the resulted Wavefront (last two dimensions) with simulation parameters
        sim_nodes_shape = self.sim_params.axes_size(axs=('H', 'W'))

        if not wavefront.size()[-2:] == sim_nodes_shape:
            warnings.warn(
                message='A shape of a resulted Wavefront does not match with SimulationParameters!'
            )

    def __len__(self):
        return len(self.init_ds)

    def __getitem__(self, ind: int) -> tuple:
        """
        Parameters
        ----------
        ind : int
            Index of element to return.

        Returns
        -------
        tuple
            An element of dataset: tuple(Wavefront, class)
            A size of a wavefront must be in a correspondence with simulation parameters!
        """
        raw_image, label = self.init_ds[ind]
        # apply transformations
        wavefront_image = self.transformations(raw_image)

        if self.target == 'label':
            return wavefront_image, label

        if self.target == 'detector':
            if self.detector_mask is None:  # no detector mask provided
                warnings.warn(
                    message='No Detector mask provided to generate targets!'
                )
            else:
                detector_image = torch.where(label == self.detector_mask, 1.0, 0.0)
                return wavefront_image, detector_image

