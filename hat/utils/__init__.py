from box.exceptions import BoxValueError
import yaml
from ensure import ensure_annotations
from box import ConfigBox
import glob
import os
import tqdm
from pathlib import Path
import cv2
import imghdr

def is_image(file_path):
    image_type = imghdr.what(file_path)
    return image_type is not None

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            # logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def resize_folder_image(input_path,output_path,size = (256,256)):
    os.makedirs(output_path,exist_ok=True)
    image_paths = glob.glob(os.path.join(input_path,"*"))
    
    for file_path in tqdm.tqdm(image_paths):
        if is_image(file_path):
            file_name = file_path.split(os.sep)[-1]
            image = cv2.imread(file_path)
            image = cv2.resize(image,size)
            new_path = os.path.join(output_path,file_name)
            cv2.imwrite(new_path,image)


def destroy_folder_image(input_path,output_path,size = (64,64)):

    os.makedirs(output_path,exist_ok=True)
    image_paths = glob.glob(os.path.join(input_path,"*"))
    
    for file_path in tqdm.tqdm(image_paths):
        if is_image(file_path):
            file_name = file_path.split(os.sep)[-1]
        
            image = cv2.imread(file_path)
            image = cv2.GaussianBlur(image, (21,21), 1)
            
            image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
            
            new_path = os.path.join(output_path,file_name)
            cv2.imwrite(new_path,image, [cv2.IMWRITE_JPEG_QUALITY, 60] )









