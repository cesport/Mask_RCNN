# The Script is modified to train on custom Labelme Data.

import os
import sys
import json
import wget
import datetime
import numpy as np
import skimage.draw
 
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
print(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize

from PIL import Image, ImageDraw

# In case of configuration
# git clone the repo and append the path Mask_RCNN repo to use the custom configurations
# git clone https://github.com/matterport/Mask_RCNN
# sys.path.append("path_to_mask_rcnn")

# Download Link 
# https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
if not os.path.isfile("mask_rcnn_coco.h5"):
    url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"
    print("Downloading the mask_rcnn_coco.h5")
    filename = wget.download(url)
    print(f"[INFO]: Download complete >> {filename}")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
 
# Change it for your dataset's name
source="Dataset2021"
############################################################
#  My Model Configurations (which you should change for your own task)
############################################################
 
class ModelConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ImageSegmentationMaskRCNN"
 
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2 # 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # Background,
    # typically after labeled, class can be set from Dataset class
    # if you want to test your model, better set it corectly based on your trainning dataset
 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500
 
    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5
 
class InferenceConfig(ModelConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
############################################################
#  Dataset (My labelme dataset loader)
############################################################
 
class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
 
    def load_mask(self,image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids
 
    def image_reference(self,image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == source:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
 
 
def train(dataset_train, dataset_val, model):
    """Train the model."""
    # Training dataset.
    dataset_train.prepare()
 
    # Validation dataset
    dataset_val.prepare()
 
    # *** This training schedule is an example. Update to your needs ***
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')
 
def test(model, image_path = None, video_path=None, savedfile=None):
    assert image_path or video_path
 
     # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Colorful
        import matplotlib.pyplot as plt
        
        _, ax = plt.subplots()
        visualize.get_display_instances_pic(image, boxes=r['rois'], masks=r['masks'], 
            class_ids = r['class_ids'], class_number=model.config.NUM_CLASSES,ax = ax,
            class_names=None,scores=None, show_mask=True, show_bbox=True)
        # Save output
        if savedfile == None:
            file_name = "test_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        else:
            file_name = savedfile
        plt.savefig(file_name)
        #skimage.io.imsave(file_name, testresult)
    elif video_path:
        pass
    print("Saved to ", file_name)
 
                
############################################################
#  Training and Validating
############################################################
 
if __name__ == '__main__':
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of your dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco', 'last' or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=./logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to test and color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to test and color splash effect on')
    parser.add_argument('--classnum', required=False,
                        metavar="class number of your detect model",
                        help="Class number of your detector.")
    args = parser.parse_args()
 
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video or args.classnum, \
            "Provide --image or --video and  --classnum of your model to apply testing"
 
 
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
 
    # Configurations
    if args.command == "train":
        config = ModelConfig()
        dataset_train, dataset_val = LabelmeDataset(), LabelmeDataset()
        dataset_train.load_labelme(args.dataset,"train")
        dataset_val.load_labelme(args.dataset,"val")
        config.NUM_CLASSES = len(dataset_train.class_info)
    elif args.command == "test":
        config = InferenceConfig()
        config.NUM_CLASSES = int(args.classnum)+1 # add backgrouond
        
    config.display()
 
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
 
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights
 
    # Load weights
    print("Loading weights ", weights_path)
    if args.command == "train":
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes if we change the backbone?
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
        # Train or evaluate
        train(dataset_train, dataset_val, model)
    elif args.command == "test":
        # we test all models trained on the dataset in different stage
        print(os.getcwd())
        filenames = os.listdir(args.weights)
        for filename in filenames:
            if filename.endswith(".h5"):
                print(f"Load weights from {filename}")
                model.load_weights(os.path.join(args.weights,filename),by_name=True)
                savedfile_name = os.path.splitext(filename)[0] + ".jpg"
                test(model, image_path=args.image,video_path=args.video, savedfile=savedfile_name)
    else:
        print("'{}' is not recognized.Use 'train' or 'test'".format(args.command))
