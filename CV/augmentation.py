from torchvision.transforms import transforms
from enum import Enum
import os
import torch
import torch.nn as nn
import numpy as np
import math
import tqdm
from traditional_aug import TraditionalAugmentation, cifar10_aug_policy
from imbalanced_dataset import get_dataset, DatasetWrapper
from torch.utils.data import DataLoader
from PIL import Image


def random_erasing(x):
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing()])
    return transform(x)


class Mixup(object):
    def __init__(self, alpha=1.0, device='cpu'):
        """Returns mixed inputs
                from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
                alpha = 1 for cifar10/100
            :param x: batch data
            :param y:
            :param alpha:
            :param device:
            :return:
            """
        self.alpha = alpha
        self.device = device

    def __call__(self, x, y=None):
        alpha = self.alpha
        device = self.device
        if y is not None and not torch.all(y.eq(y[0])):
            raise Exception('data label inconsistent')

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index1 = torch.randperm(batch_size).to(device)
        index2 = torch.cat([index1[1:batch_size], index1[0].reshape(-1)], dim=0)
        if not torch.all(~index1.eq(index2)):
            print("index overlap")

        mixed_x = lam * x[index1, :] + (1 - lam) * x[index2, :]

        return mixed_x


def rand_bbox(size, lam):
    """ Code from  https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    """
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Cutmix(object):
    def __init__(self, beta, device):
        """ Returns cutmixed inputs,
                Code from  https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
        :param beta: 1.0 for cifar10
        :param device:
        """
        self.beta = beta
        self.device = device

    def __call__(self, x, device=None):
        """
        :param x: batch data
        :return:
        """
        d = device if device is not None else self.device
        if self.beta > 0 :
            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(x.size()[0]).to(d)
            index2 = torch.cat([rand_index[1:x.size()[0]], rand_index[0].reshape(-1)], dim=0)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[index2, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        return x


class Grid(object):
    """
        Code from https://github.com/dvlab-research/GridMask/blob/master/imagenet_grid/utils/grid.py
    """
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode

    def __call__(self, img, device):
        h = img.size(1)
        w = img.size(2)

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        mask = torch.from_numpy(mask).float().to(device)
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GridMask(nn.Module):
    """
        Code from https://github.com/dvlab-research/GridMask/blob/master/imagenet_grid/utils/grid.py
    """
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, device='cpu'):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.grid = Grid(d1, d2, rotate, ratio, mode)
        self.device = device

    def __call__(self, x, device=None):
        d = device if device is not None else self.device
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i], d))
        y = torch.cat(y).view(n, c, h, w)
        return y


# class AugMixOp(object):
#     def __init__(self):
#         self.IMAGE_SIZE = 32
#         self.augmentations = [
#             self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
#             self.translate_x, self.translate_y
#         ]
#
#         self.augmentations_all = [
#             self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
#             self.translate_x, self.translate_y, self.color, self.contrast, self.brightness, self.sharpness
#         ]
#
#     @staticmethod
#     def int_parameter(level, maxval):
#         """Helper function to scale `val` between 0 and maxval .
#         Args:
#           level: Level of the operation that will be between [0, `PARAMETER_MAX`].
#           maxval: Maximum value that the operation can have. This will be scaled to
#             level/PARAMETER_MAX.
#         Returns:
#           An int that results from scaling `maxval` according to `level`.
#         """
#         return int(level * maxval / 10)
#
#     @staticmethod
#     def float_parameter(level, maxval):
#         """Helper function to scale `val` between 0 and maxval.
#         Args:
#           level: Level of the operation that will be between [0, `PARAMETER_MAX`].
#           maxval: Maximum value that the operation can have. This will be scaled to
#             level/PARAMETER_MAX.
#         Returns:
#           A float that results from scaling `maxval` according to `level`.
#         """
#         return float(level) * maxval / 10.
#
#     @staticmethod
#     def sample_level(n):
#         return np.random.uniform(low=0.1, high=n)
#
#     @staticmethod
#     def autocontrast(pil_img, _):
#         return ImageOps.autocontrast(pil_img)
#
#     @staticmethod
#     def equalize(pil_img, _):
#         return ImageOps.equalize(pil_img)
#
#     @staticmethod
#     def posterize(pil_img, level):
#         level = AugMixOp.int_parameter(AugMixOp.sample_level(level), 4)
#         return ImageOps.posterize(pil_img, 4 - level)
#
#     @staticmethod
#     def rotate(pil_img, level):
#         degrees = AugMixOp.int_parameter(AugMixOp.sample_level(level), 30)
#         if np.random.uniform() > 0.5:
#             degrees = -degrees
#         return pil_img.rotate(degrees, resample=Image.BILINEAR)
#
#     @staticmethod
#     def solarize(pil_img, level):
#         level = AugMixOp.int_parameter(AugMixOp.sample_level(level), 256)
#         return ImageOps.solarize(pil_img, 256 - level)
#
#     def shear_x(self, pil_img, level):
#         level = AugMixOp.float_parameter(AugMixOp.sample_level(level), 0.3)
#         if np.random.uniform() > 0.5:
#             level = -level
#         return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
#                                  Image.AFFINE, (1, level, 0, 0, 1, 0),
#                                  resample=Image.BILINEAR)
#
#     def shear_y(self, pil_img, level):
#         level = self.float_parameter(self.sample_level(level), 0.3)
#         if np.random.uniform() > 0.5:
#             level = -level
#         return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
#                                  Image.AFFINE, (1, 0, 0, level, 1, 0),
#                                  resample=Image.BILINEAR)
#
#     def translate_x(self, pil_img, level):
#         level = self.int_parameter(self.sample_level(level), self.IMAGE_SIZE / 3)
#         if np.random.random() > 0.5:
#             level = -level
#         return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
#                                  Image.AFFINE, (1, 0, level, 0, 1, 0),
#                                  resample=Image.BILINEAR)
#
#     def translate_y(self, pil_img, level):
#         level = self.int_parameter(self.sample_level(level), self.IMAGE_SIZE / 3)
#         if np.random.random() > 0.5:
#             level = -level
#         return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
#                                  Image.AFFINE, (1, 0, 0, 0, 1, level),
#                                  resample=Image.BILINEAR)
#
#     @staticmethod
#     # operation that overlaps with ImageNet-C's test set
#     def color(pil_img, level):
#         level = AugMixOp.float_parameter(AugMixOp.sample_level(level), 1.8) + 0.1
#         return ImageEnhance.Color(pil_img).enhance(level)
#
#     @staticmethod
#     # operation that overlaps with ImageNet-C's test set
#     def contrast(pil_img, level):
#         level = AugMixOp.float_parameter(AugMixOp.sample_level(level), 1.8) + 0.1
#         return ImageEnhance.Contrast(pil_img).enhance(level)
#
#     @staticmethod
#     # operation that overlaps with ImageNet-C's test set
#     def brightness(pil_img, level):
#         level = AugMixOp.float_parameter(AugMixOp.sample_level(level), 1.8) + 0.1
#         return ImageEnhance.Brightness(pil_img).enhance(level)
#
#     @staticmethod
#     # operation that overlaps with ImageNet-C's test set
#     def sharpness(pil_img, level):
#         level = AugMixOp.float_parameter(AugMixOp.sample_level(level), 1.8) + 0.1
#         return ImageEnhance.Sharpness(pil_img).enhance(level)
#
#     @staticmethod
#     def normalize(image):
#         # CIFAR-10 constants
#         MEAN = [0.4914, 0.4822, 0.4465]
#         STD = [0.2023, 0.1994, 0.2010]
#         """Normalize input image channel-wise to zero mean and unit variance."""
#         image = image.transpose(2, 0, 1)  # Switch to channel-first
#         mean, std = np.array(MEAN), np.array(STD)
#         image = (image - mean[:, None, None]) / std[:, None, None]
#         return image.transpose(1, 2, 0)
#
#     @staticmethod
#     def apply_op(image, op, severity):
#       image = np.clip(image * 255., 0, 255).astype(np.uint8)
#       pil_img = Image.fromarray(image)  # Convert to PIL.Image
#       pil_img = op(pil_img, severity)
#       return np.asarray(pil_img) / 255.
#
#     def augment_and_mix(self, image, severity=3, width=3, depth=-1, alpha=1.):
#       """Perform AugMix augmentations and compute mixture.
#       Args:
#         image: Raw input image as float32 np.ndarray of shape (h, w, c)
#         severity: Severity of underlying augmentation operators (between 1 to 10).
#         width: Width of augmentation chain
#         depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
#           from [1, 3]
#         alpha: Probability coefficient for Beta and Dirichlet distributions.
#       Returns:
#         mixed: Augmented and mixed image.
#       """
#
#       ws = np.float32(
#           np.random.dirichlet([alpha] * width))
#       m = np.float32(np.random.beta(alpha, alpha))
#
#       mix = np.zeros_like(image)
#       for i in range(width):
#         image_aug = image.copy()
#         d = depth if depth > 0 else np.random.randint(1, 4)
#         for _ in range(d):
#           op = np.random.choice(self.augmentations)
#           image_aug = self.apply_op(image_aug, op, severity)
#         # Preprocessing commutes since all coefficients are convex
#         mix += ws[i] * self.normalize(image_aug)
#
#       mixed = (1 - m) * self.normalize(image) + m * mix
#       return mixed


class AugmentType(Enum):
    CutMix=1
    MixUp=2
    GridMask=3
    TraditionalAug=4


def convert_tensor_to_PILimages(images_tensor):
    tt = transforms.ToPILImage()
    n = images_tensor.shape[0]
    result = []
    for i in range(n):
        result.append(tt(images_tensor[i]).convert('RGB'))
    return result


class CVAugment(object):
    def __init__(self, device='cpu'):
        self.aug_policy = {}
        self.__set_up_policy()
        self.device = device
        # the amount of generated(augmented) images for each image
        self.expand_times = 0
        for at in AugmentType:
            self.expand_times += self.aug_policy[at]

    def __set_up_policy(self):
        """ Set the number of times each augmentation policy is used (for each sample)
        """
        self.aug_policy[AugmentType.CutMix] = 5
        self.aug_policy[AugmentType.MixUp] = 5
        # self.aug_policy[AugmentType.RandomErasing] = 2
        self.aug_policy[AugmentType.GridMask] = 5
        self.aug_policy[AugmentType.TraditionalAug] = 20

    def get_aug(self, aug_method):
        """ Return augmentation operator object by aug_method
        """
        aug = None
        if aug_method == AugmentType.MixUp:
            aug = Mixup(alpha=1.0, device=self.device)
        elif aug_method == AugmentType.CutMix:
            aug = Cutmix(beta=1.0, device=self.device)
        elif aug_method == AugmentType.GridMask:
            aug = GridMask(d1=24, d2=33, rotate=1, ratio=0.4, mode=1, device=self.device)
        elif aug_method == AugmentType.TraditionalAug:
            aug = TraditionalAugmentation(policies=cifar10_aug_policy())
        else:
            raise Exception("No such "+aug_method)
        return aug

    def generate_augmented_data(self, origin_dst:DatasetWrapper, batch_size=64, save_path=None):
        """ Generate auxiliary dataset for source set by data augmentations

        :param origin_dst: whole train set(Source Set) in imbalanced-cifar-10
        :param batch_size: batch size in DataLoader
        :param save_path: the folder path to save generated(augmented) images(Auxiliary Dataset)
        :return:
        """
        bar = tqdm.tqdm(total=self.expand_times*len(origin_dst.class_split_indexes.keys()))

        for label in origin_dst.class_split_indexes.keys():
            class_x = origin_dst.get_dataset_by_class(label)
            class_loader = DataLoader(class_x, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            class_save_path = save_path+str(label)+'/'
            if not os.path.exists(class_save_path):
                os.makedirs(class_save_path)
            save_index = 0
            for at in AugmentType:
                augmentor = self.get_aug(at)
                for t in range(self.aug_policy[at]):
                    for imgs, _ in class_loader:
                        imgs = imgs.to(self.device)
                        augmented_PIL_images = convert_tensor_to_PILimages(augmentor(imgs))
                        for i, pil_img in enumerate(augmented_PIL_images):
                            pil_img.save(class_save_path + str(save_index+i) + '.jpg')
                        save_index += len(augmented_PIL_images)
                    bar.update(1)
        bar.close()


def plot_test(imgs, row_title=None, **imshow_kwargs):
    from PIL import Image
    import matplotlib.pyplot as plt
    import torch
    plt.rcParams["savefig.bbox"] = 'tight'
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_cols*5, num_rows*5))
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[0, 0].set(title='Original image')
    axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


def cvaugmentor_test(trainset):
    """ Testing different data augmentation strategies and visualizing augmented samples

    :param trainset:
    """
    loader = DataLoader(trainset, batch_size=4)
    augmentor = CVAugment(device='cuda:0')
    aug_images = []
    for imgs, _ in loader:
        origin_images = imgs.to('cuda:0')
        aug_images.extend(convert_tensor_to_PILimages(origin_images))
        plot_test(aug_images)

        aug = augmentor.get_aug(AugmentType.TraditionalAug)
        aug_images.extend(convert_tensor_to_PILimages(aug(origin_images)))

        aug = augmentor.get_aug(AugmentType.GridMask)
        aug_images.extend(convert_tensor_to_PILimages(aug(origin_images)))

        aug = augmentor.get_aug(AugmentType.MixUp)
        aug_images.extend(convert_tensor_to_PILimages(aug(origin_images)))

        aug = augmentor.get_aug(AugmentType.CutMix)
        aug_images.extend(convert_tensor_to_PILimages(aug(origin_images)))

        plot_test(aug_images)
        break

    path = '/ssd1/omf/datasets/cifar10_pool/'
    for i, img in enumerate(aug_images):
        img.save(path + str(i) + '.jpg')

    saved_images = []
    for i in range(len(aug_images)):
        saved_images.append(Image.open(path + str(i) + '.jpg'))
    print(saved_images[0].size)
    plot_test(saved_images)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = get_dataset(dst_name='im_cifar10', split='train')
    trainset.update_transform(transform)
    # cvaugmentor_test(trainset)
    augmentor = CVAugment(device='cuda:0')
    augmentor.generate_augmented_data(trainset, save_path='/ssd1/omf/datasets/cifar10_pool/')

