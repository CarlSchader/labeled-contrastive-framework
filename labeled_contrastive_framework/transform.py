from torchvision import transforms
# from transformers import AutoImageProcessor

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_transform(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
    is_tensor=False,
):

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    if is_tensor:
        transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform

def make_eval_transform(
    crop_size=224,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1])
    ])
    return transform

# imageProcessor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
#
# def make_eval_transform(
#     crop_size=224,
#     normalization=((0.485, 0.456, 0.406),
#                    (0.229, 0.224, 0.225))
# ):
#
#     return transforms.Lambda(lambda x: imageProcessor(x, return_tensors='pt').pixel_values.reshape(3, 224, 224))
#     

if __name__ == '__main__':
    import torch, argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True)

    args = parser.parse_args()

    test_input = Image.open(args.image)

    transform = make_eval_transform()
    print(transform(test_input).shape)

