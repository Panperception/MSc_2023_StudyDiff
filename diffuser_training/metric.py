from log.app_logger import logger

def diffusion_metric(result, epoch_size=2, batch_size=50, generate_img_dir=''):

    result['generate_epoch_size'] = epoch_size
    result['generate_batch_size'] = batch_size

    from datasets import load_from_disk

    dataset = load_from_disk(result['dataset_dir'])
    dataset = dataset['train']

    import numpy as np

    def transform(examples):

        images = []
        for image in examples['image']:
            image = image.convert('RGB')
            image = image.resize((128, 128))
            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))
            images.append(image)

        return {"images": images}

    dataset.set_transform(transform)

    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=64)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for item in train_dataloader:
        images = item['images']
        fid.update(images, real=True)

    from diffusers import DDPMPipeline
    from datetime import datetime

    t = DDPMPipeline.from_pretrained('ddpm-butterflies-128')
    t.to('cuda')

    start = datetime.now()
    generated_image = []
    for i in range(epoch_size):
        images = t(batch_size=batch_size, num_inference_steps=result['num_inference_steps']).images
        generated_image.extend(images)
    end = datetime.now()
    result['generate_time'] = str(end - start)

    from torchvision import transforms
    from torchmetrics.image.inception import InceptionScore

    inception = InceptionScore()
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    i = 0
    for image in generated_image:
        image.save(generate_img_dir.format(i))
        i = i + 1
        image = preprocess(image)
        image = image * 255
        image = image.to(torch.uint8)
        image = image.unsqueeze(0)
        fid.update(image, real=False)
        inception.update(image)

    result['fid'] = fid.compute().item()
    inception_score = inception.compute()
    result['is_mean'] = inception_score[0].item()
    result['is_sd'] = inception_score[1].item()

    return result