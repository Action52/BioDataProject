import yaml


def load_config(file_path):
    """
    Load a YAML file.

    Args:
    file_path (str): Path to the YAML file.

    Returns:
    dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
