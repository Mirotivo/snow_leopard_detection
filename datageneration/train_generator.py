# %%
import os
import csv
import torch
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

dataset_images_count = 200


# define paths
data_dir = 'synthetic_snow_leopard_200'
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')

# set random seeds for reproducibility
seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda:0")


# Define the possible values for each parameter
poses = ['standing', 'walking', 'running', 'lying down', 'crouching']
backgrounds = ['snowy mountains', 'rocky terrain', 'grassy meadows', 'zoo']
interactions = ['hunting prey', 'playing with other snow leopards', 'interacting with environment']
weather_conditions = ['snowstorm', 'rain', 'fog', 'sunny']
camera_parameters = ['low aperture', 'fast shutter speed', 'short focal length']
occlusions = ['tree', 'rock', 'other objects']
camera_viewpoints = ['front view', 'side view', 'top view', 'back view', 'camera trap']
snow_depths = ['deep', 'shallow', 'none']
wind_speeds = ['strong', 'moderate', 'weak']
temperatures = ['cold', 'very cold', 'extremely cold', 'warm']
altitudes = ['high', 'low']
sun_positions = ['morning', 'midday', 'afternoon', 'night']
moon_phases = ['full moon', 'half moon', 'new moon']
animal_presences = ['none', 'other snow leopards', 'prey animals']
distances = ['close', 'far']
behaviors = ['marking territory', 'vocalizing', 'grooming']

# create required folders if they don't exist
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(annotations_dir):
    os.mkdir(annotations_dir)
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

with open(os.path.join(annotations_dir, 'dataset.csv'), 'w', newline='') as csvfile:
    fieldnames = ['prompt', 'file_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(dataset_images_count):

        # Generate random values for each parameter
        pose = random.choice(poses)
        background = random.choice(backgrounds)
        interaction = random.choice(interactions)
        weather_condition = random.choice(weather_conditions)
        camera_parameter = random.choice(camera_parameters)
        occlusion = random.choice(occlusions)
        camera_viewpoint = random.choice(camera_viewpoints)
        snow_depth = random.choice(snow_depths)
        wind_speed = random.choice(wind_speeds)
        temperature = random.choice(temperatures)
        altitude = random.choice(altitudes)
        sun_position = random.choice(sun_positions)
        moon_phase = random.choice(moon_phases)
        animal_presence = random.choice(animal_presences)
        distance = random.choice(distances)
        behavior = random.choice(behaviors)

        prompt = f"A snow leopard is {pose} in a {background} environment, {interaction}. The weather is {weather_condition} and the temperature is {temperature}. The camera has {camera_parameter} and is positioned at a {camera_viewpoint}. The snow leopard is partially obscured by a {occlusion}, and the camera is capturing the scene from {distance} distance. The snow depth is {snow_depth} and the wind is {wind_speed}. The altitude is {altitude} and the sun is in the {sun_position}. The moon is in the {moon_phase}. There are {animal_presence} nearby. The snow leopard is {behavior}."
        image = pipe(prompt).images[0]

        # save image with random number in the file name
        file_name = f"snow_leopard_{random.randint(0, 10000)}.png"
        image.save(os.path.join(images_dir, file_name))

        writer.writerow({'prompt': prompt, 'file_name': file_name})
        print(f"Saved image {file_name}")

# %%
