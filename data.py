from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import imfusion as imf


class PatchDataset(Dataset):
    """
    Dataset class for training the model
    The dataset is preprocessed using imfusion package and yields patches of a specified or variable size
    Parses a data list in .txt format with tab separated columns in the format:
    #DataPath   LabelPath
    file_path_ct_patient_0  file_path_lbl_patient_0
    ...
    file_path_ct_patient_n  file_path_lbl_patient_n

    Args:
        data_file (str): Path to the data list file
        n_downsample (int): Number of downsampling layers in the model
        spacing (list): Spacing of the images
        n_labels (int): Number of labels in the segmentation task (without background)
        patch_size_range (list): Range of patch sizes to sample from
        use_fixed_patch_size (bool): Whether to use fixed patch size or not
        fixed_patch_size (list): Fixed patch size to use if use_fixed_patch_size is True
        only_divisible_patch_sizes (bool): Whether to only use patch sizes that are divisible by 2**n_downsample
        n_dim (int): Number of dimensions in the images (2 or 3)
    """

    def __init__(
        self,
        data_file,
        n_downsample,
        spacing=[2.0, 2.0, 2.0],
        n_labels=1,
        patch_size_range=[64, 256],
        use_fixed_patch_size=False,
        fixed_patch_size=None,
        only_divisible_patch_sizes=False,  # TODO: remove? I only reported the results on divisible patch sizes
        n_dim=3,
    ):

        # Read data file
        with open(data_file) as f:
            data_list = f.read()
            data_list = data_list.split("\n")
        data_list = [x.split("\t") for x in data_list[1:] if x != ""]
        data_list = np.array(data_list)

        self.im_files = data_list[:, 0]
        self.lbl_files = data_list[:, 1]

        self.im_files = self.im_files.flatten()
        self.lbl_files = self.lbl_files.flatten()

        # Specific to U-Net AP
        self.n_downsample = n_downsample
        min_patch_size = patch_size_range[0]
        max_patch_size = patch_size_range[1]
        self.patch_size_list = (
            list(range(min_patch_size, max_patch_size, 2**self.n_downsample))
            if only_divisible_patch_sizes
            else list(range(min_patch_size, max_patch_size))
        )

        self.spacing = spacing
        self.ct_window = [-155, 245]  # window level
        self.background_ct = -1024

        assert n_dim in [2, 3], f"Invalid number of dimensions: {n_dim}"
        self.data_dim = n_dim

        self.one_hot = np.eye(n_labels + 1)
        self.label_values = list(range(1, n_labels + 1))

        # Specific to U-Net FP
        self.use_fixed_patch_size = use_fixed_patch_size
        self.fixed_patch_size = fixed_patch_size

        # imfusion preprocessing operations
        self.im_preprocess_ops = [
            imf.machinelearning.MakeFloatOperation(),
            imf.machinelearning.BakeTransformationOperation(),
            imf.machinelearning.ResampleOperation(resolution=self.spacing),
        ]
        self.lbl_preprocess_ops = [
            imf.machinelearning.SetLabelModalityOperation(),
            imf.machinelearning.BakeTransformationOperation(),
            imf.machinelearning.ResampleOperation(resolution=self.spacing),
        ]

    def imfusion_preprocess(
        self, im: imf.SharedImageSet, lbl: imf.SharedImageSet
    ) -> tuple[imf.SharedImageSet, imf.SharedImageSet]:
        """
        Preprocess the images and labels using imfusion operations
        Docs: https://teamcity.imfusion.com/repository/download/ImFusionSuite_CiMasterPullRequestsDockerGcc940Ubuntu2004/.lastSuccessful/PythonDoc.zip!//ml_op_bindings.html
        Args:
            im (numpy.ndarray): Image to preprocess
            lbl (numpy.ndarray): Label to preprocess
        """
        # TODO: add link to imfusion preprocessing operations and samplers documentation sin docstring?
        for op in self.im_preprocess_ops:
            im = op.process(im)

        for op in self.lbl_preprocess_ops:
            lbl = op.process(lbl)

        return im, lbl

    def imfusion_sampler(
        self, im: imf.SharedImageSet, lbl: imf.SharedImageSet, roi_size: np.ndarray, p: float = 0.5
    ) -> tuple[imf.SharedImageSet, imf.SharedImageSet]:
        """
        Sample a random ROI from the image and label using imfusion samplers
        Docs: https://teamcity.imfusion.com/repository/download/ImFusionSuite_CiMasterPullRequestsDockerGcc940Ubuntu2004/.lastSuccessful/PythonDoc.zip!//ml_samplers_bindings.html
        Args:
            im (numpy.ndarray): Image to sample from
            lbl (numpy.ndarray): Label to sample from
            roi_size (numpy.ndarray): Size of the ROI
            p (float): Probability of using LabelROISampler
        """
        if np.random.rand() < p:
            sampler = imf.machinelearning.LabelROISampler(
                roi_size=roi_size,
                labels_values=self.label_values,
                sample_boundaries_only=False,
            )
        else:
            sampler = imf.machinelearning.RandomROISampler(roi_size=roi_size)
        roi = sampler.compute_roi(lbl)
        im = sampler.extract_roi(im, roi)
        lbl = sampler.extract_roi(lbl, roi)
        return im, lbl

    def make_size_divisible(
        self, im: np.ndarray, lbl: np.ndarray, patch_size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make the image and label size divisible by 2**n_downsample
        Args:
            im (numpy.ndarray): Image to pad
            lbl (numpy.ndarray): Label to pad
            patch_size (numpy.ndarray): Size of the patch
        """
        for i in range(self.data_dim):
            if patch_size[i] % (2**self.n_downsample) != 0:
                pad = 2**self.n_downsample - (patch_size[i] % (2**self.n_downsample))
                pad_left = pad // 2
                pad_right = pad - pad_left
                padding = [[0, 0] for _ in range(self.data_dim)]
                padding[i][0] = pad_left
                padding[i][1] = pad_right
                im = np.pad(im, padding, mode="edge")
                lbl = np.pad(lbl, padding, mode="edge")

        return im, lbl

    def __getitem__(self, index: int) -> dict:
        im = imf.load(self.im_files[index])[0]
        lbl = imf.load(self.lbl_files[index])[0]

        im, lbl = self.imfusion_preprocess(im, lbl)

        if self.use_fixed_patch_size:  # U-Net FP
            patch_size = np.array(self.fixed_patch_size)
        else:  # U-Net AP
            ps_x = np.random.choice(self.patch_size_list)
            ps_y = np.random.choice(self.patch_size_list)
            ps_z = np.random.choice(self.patch_size_list)
            patch_size = np.array([ps_x, ps_y, ps_z])

        im, lbl = self.imfusion_sampler(im, lbl, patch_size, p=0.3)
        im = im.numpy().squeeze()
        lbl = lbl.numpy().squeeze()

        # depth, height, width to be coherent with input tensor shape
        patch_size = patch_size[::-1]

        # Pad the image to be divisible by 2**n_downsample
        im, lbl = self.make_size_divisible(im, lbl, patch_size)

        # Clip and normalize the image
        im = np.clip(im, self.ct_window[0], self.ct_window[1])
        im = (im - self.ct_window[0]) / (self.ct_window[1] - self.ct_window[0])

        # Add channel dimension
        im = im[None, ...]
        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2)

        identifier = Path(self.im_files[index]).name.replace(".nii.gz", "")

        return {
            "im": im.astype(np.float32),
            "lbl": lbl.astype(np.float32),
            "ps": patch_size.astype(np.float32),
            "identifier": identifier,
        }

    def __len__(self):
        return len(self.im_files)


class BasicDataset(Dataset):
    """
    Dataset class for performing inference
    The dataset is preprocessed using imfusion package and yields whole images
    Parses a data list in .txt format with tab separated columns in the format:
    #DataPath   LabelPath
    file_path_ct_patient_0  file_path_lbl_patient_0
    ...
    file_path_ct_patient_n  file_path_lbl_patient_n

    Args:
        data_file (str): Path to the data list file
        spacing (list): Spacing of the images
        n_labels (int): Number of labels in the segmentation task (without background)
        flip_image_content (bool): Whether to flip the image content or not (only applicable to Learn2Reg dataset)
        flip_axes (list): Axes to flip the image content (only applicable to Learn2Reg dataset)
        n_dim (int): Number of dimensions in the images (2 or 3)
    """

    def __init__(
        self, data_file, spacing=[2.0, 2.0, 2.0], n_labels=1, n_dim=3, flip_image_content=False, flip_axes=None
    ):
        # Read data file
        with open(data_file) as f:
            data_list = f.read()
            data_list = data_list.split("\n")
        data_list = [x.split("\t") for x in data_list[1:] if x != ""]
        data_list = np.array(data_list)

        self.im_files = data_list[:, 0]
        self.lbl_files = data_list[:, 1]

        self.im_files = self.im_files.flatten()
        self.lbl_files = self.lbl_files.flatten()

        self.spacing = spacing
        self.ct_window = [-155, 245]
        self.background_ct = -1024
        self.one_hot = np.eye(n_labels + 1)
        self.data_dim = n_dim

        # Parameter specific to Learn2Reg
        self.flip_image_content = flip_image_content
        self.flip_axes = flip_axes

        # Imfusion preprocessing operations
        self.im_preprocess_ops = [
            imf.machinelearning.MakeFloatOperation(),
            imf.machinelearning.BakeTransformationOperation(),
            imf.machinelearning.ResampleOperation(resolution=self.spacing),
        ]
        self.lbl_preprocess_ops = [
            imf.machinelearning.SetLabelModalityOperation(),
            imf.machinelearning.BakeTransformationOperation(),
            imf.machinelearning.ResampleOperation(resolution=self.spacing),
        ]

    def imfusion_preprocess(
        self, im: imf.SharedImageSet, lbl: imf.SharedImageSet
    ) -> tuple[imf.SharedImageSet, imf.SharedImageSet]:
        """
        Preprocess the images and labels using imfusion operations
        Args:
            im (numpy.ndarray): Image to preprocess
            lbl (numpy.ndarray): Label to preprocess
        """
        for op in self.im_preprocess_ops:
            im = op.process(im)

        for op in self.lbl_preprocess_ops:
            lbl = op.process(lbl)

        return im, lbl

    def __getitem__(self, index: int) -> dict:
        im = imf.load(self.im_files[index])[0]
        lbl = imf.load(self.lbl_files[index])[0]

        im, lbl = self.imfusion_preprocess(im, lbl)
        im = im.numpy().squeeze()
        lbl = lbl.numpy().squeeze()

        # Only necessary for Learn2Reg dataset
        if self.flip_image_content:
            im = np.flip(im, axis=self.flip_axes)
            lbl = np.flip(lbl, axis=self.flip_axes)

        # Clip and normalize the image
        im = np.clip(im, self.ct_window[0], self.ct_window[1])
        im = (im - self.ct_window[0]) / (self.ct_window[1] - self.ct_window[0])

        # Add channel dimension
        im = im[None, ...]
        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2)

        identifier = Path(self.im_files[index]).name.replace(".nii.gz", "")

        return {
            "im": im.astype(np.float32),
            "lbl": lbl.astype(np.float32),
            "identifier": identifier,
        }

    def __len__(self):
        return len(self.im_files)
