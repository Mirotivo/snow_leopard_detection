# %%
import os
import csv
import urllib.request


# open the CSV file containing the image URLs
observations = [
    ('snow_leopard', '305347')
]

# define paths
data_dir = 'inaturalist'
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')

# create required folders if they don't exist
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(annotations_dir):
    os.mkdir(annotations_dir)
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

for animal, obs_id in observations:
    with open(os.path.join(annotations_dir, f'{animal}_observations-{obs_id}.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # loop through each row in the CSV file
        for row in reader:
            # extract the image URL and generate the file_name
            image_url = row['image_url']
            file_name = f'{animal}_{image_url.split("/")[-2]}.jpg'

            try:
                urllib.request.urlretrieve(image_url, os.path.join(images_dir, file_name))
            except (urllib.error.URLError, urllib.error.HTTPError):
                print('{} failure'.format(image_url))
            else:
                print('{} saved.'.format(file_name))

# %%
