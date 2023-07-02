import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify the directory of your dataset
data_dir = r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\ELECTRONIC FORGERY'

# Set the desired height and width for your images
height = 113
width = 250

# Create an ImageDataGenerator object with augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# Specify the target directory to save the augmented images
output_dir = r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\AUGMENTED ELECTRONIC FORGERY'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of file names in the original dataset directory
original_file_names = os.listdir(data_dir)

# Sort the file names in ascending order
original_file_names.sort()

# Generate augmented images and save them to the output directory
person = 1
start_number = 6
for i, file_name in enumerate(original_file_names):
    # Construct the path to the current image file in the original dataset
    img_path = os.path.join(data_dir, file_name)
    if file_name[0] != person:
        start_number = 6
        person = int(file_name[0])

    # Load the image using the tensorflow.keras.preprocessing module
    img = image.load_img(img_path, target_size=(height, width))

    # Convert the PIL image to a numpy array
    x = image.img_to_array(img)

    # Reshape the array to (1, height, width, channels)
    x = x.reshape((1,) + x.shape)

    # Generate augmented images and save them to the output directory
    j = 0

    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=f'{file_name[0]}.', save_format='png'):
        j += 1
        print(start_number)
        start_number += 1
        if j >= 5:
            break
