import os

from app_logger import logger


def generate_imgs(
    model, prompt, file_path, epoch_total, num_per_epoch, other_args=None
):
    prompt = [prompt] * num_per_epoch
    image_count = 0
    for i in range(epoch_total):
        logger.info(prompt[0])
        logger.info(f"current epoch: {i}, total epoch: {epoch_total}")
        if other_args is None:
            generate_images = model(prompt)["images"]
        else:
            generate_images = model(prompt, **other_args)["images"]

        for generate_image in generate_images:
            file_name = str(image_count) + ".png"
            image_count += 1
            generate_image.save(os.path.join(file_path, file_name))


from diffusers import DiffusionPipeline

logger.info("begin")
stable_diffusion_dir = r"/root/autodl-fs/pre_trained_models/runwayml-stable-diffusion-v1-5/runwayml-stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(stable_diffusion_dir)
stable_diffusion.to("cuda")
logger.info("stable diffusion load success")

classes = ("plane", "car", "ship")

num_per_epoch = 10
epoch = 1000


for current_class in classes:
    dir_path = os.path.join("/root/autodl-tmp/tmp", current_class)
    generate_imgs(stable_diffusion, current_class, dir_path, epoch, num_per_epoch)
