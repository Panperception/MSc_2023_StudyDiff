from log.app_logger import logger
from diffuser_training.diffusion_training import train_model
from diffuser_training.metric import diffusion_metric
import json

if __name__ == '__main__':

    dataset_dir_list = [
        r'/root/autodl-tmp/dataset/smithsonian_butterfly',
        r'/root/autodl-tmp/dataset/oxford_flower',
        r'/root/autodl-tmp/dataset/anime_face',
        r'/root/autodl-tmp/dataset/anime_face_all'
    ]

    for i in range(1000, 1001, 100):

        dataset_index = 3
        num_inference_steps = i

        dataset_name = dataset_dir_list[dataset_index].split('/')[-1]
        output_dir = f'/root/autodl-tmp/trained_models/{dataset_name}/{num_inference_steps}'

        my_train_config = {
            'num_epochs': 50,
            'dataset': dataset_dir_list[dataset_index],
            'num_inference_steps': num_inference_steps,
            'output_dir': output_dir
        }
        train_model(my_train_config)