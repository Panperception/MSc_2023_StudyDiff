from datasets import load_from_disk


def get_food101():
    dataset_dir = r"/root/autodl-fs/dataset/food101"
    dataset = load_from_disk(dataset_dir)

    train_dataset = dataset["train"]

    validate_test_dataset = dataset["validation"].train_test_split(
        test_size=0.5, shuffle=True
    )
    val_dataset = validate_test_dataset["train"]
    test_dataset = validate_test_dataset["test"]

    num_of_category = 101

    return train_dataset, val_dataset, test_dataset, num_of_category


def get_cat_vs_dog():
    dataset_dir = r"/root/autodl-fs/dataset/cat_vs_dog"
    dataset = load_from_disk(dataset_dir)
    dataset = dataset["train"]
    dataset = dataset.rename_column("labels", "label")
    tmp = dataset.train_test_split(test_size=0.2, shuffle=True)
    train_dataset = tmp["train"]

    tmp = tmp["test"].train_test_split(test_size=0.5, shuffle=True)
    val_dataset = tmp["train"]
    test_dataset = tmp["test"]

    num_of_category = 2

    return train_dataset, val_dataset, test_dataset, num_of_category
