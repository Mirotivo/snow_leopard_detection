## Synthetic Snow Leopard 
This script generates a synthetic dataset of snow leopards with annotations.

### How to use
1. Run the train_generator.py script
2. The dataset will be created in the synthetic_snow_leopard folder with two subfolders, images and annotations
3. Run tools/misc/images2coco.py to convert the images and annotations to COCO format
4. Annotate the images as needed

### Folder Structure
After running train_generator.py, the dataset folder structure should look like this:

```
synthetic_snow_leopard/
├── annotations/
└── images/
```

The images folder will contain the synthetic images of snow leopards, and the annotations folder will contain the corresponding annotations in JSON format.

### Image Annotation
To annotate the images, you can use any COCO-compatible annotation tool such as VGG Image Annotator (VIA) or labelme.

After annotating the images, you can use the resulting annotations in your machine learning pipeline.


### Acknowledgements
* Stable Diffusion - the pre-trained model used for generating the images.
* COCO Dataset - the annotation format used for the generated images. 
