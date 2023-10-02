from datasets import load_from_disk
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision import transforms

transform = Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


def transforms(examples):
    tmp = []
    for image in examples['image']:
        image = image.convert('L')
        image = transform(image)
        tmp.append(image)
    
    examples["pixel_values"] = tmp
    del examples["image"]

    return examples

def get_fashion_mnist():

    dataset = load_from_disk("/root/autodl-fs/dataset/fashion_mnist")
    image_size = 28
    channels = 1
    batch_size = 128

    dataset = dataset.filter(lambda x: x['label'] == 0)

    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    dataloader = DataLoader(
        transformed_dataset["train"], batch_size=batch_size, shuffle=True, num_workers=3
    )

    return image_size, channels, dataloader
