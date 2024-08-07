from PIL import Image
import numpy as np

def convert(image_path, num_sub_images, num_slices_per_image):
    img = Image.open(image_path)
    
    img_width, img_height = img.size
    sub_image_width = img_width // num_sub_images
    
    # Create a list to store the slices
    slices = [[] for _ in range(num_slices_per_image)]
    
    # Iterate over each sub-image
    for i in range(num_sub_images):
        sub_image = img.crop((i * sub_image_width, 0, (i + 1) * sub_image_width, img_height))
        
        # Calculate the width of each slice
        slice_width = sub_image_width // num_slices_per_image
        
        # Split the sub-image into slices
        for j in range(num_slices_per_image):
            slice_img = sub_image.crop((j * slice_width, 0, (j + 1) * slice_width, img_height))
            slices[j].append(slice_img)
    
    # Create new images from the slices
    new_images = []
    slice_width = sub_image_width // num_slices_per_image
    
    for i in range(num_slices_per_image):
        # Create a new image with the same height as the original and width to fit all slices
        new_img = Image.new('RGB', (num_sub_images * slice_width, img_height))
        
        # Paste all slices into the new image
        for j in range(num_sub_images):
            new_img.paste(slices[i][j], (j * slice_width, 0))
        
        new_images.append(new_img)
    
    # Combine all new images horizontally
    final_width = num_sub_images * slice_width * num_slices_per_image
    final_img = Image.new('RGB', (final_width, img_height))
    
    for i, new_img in enumerate(reversed(new_images)):
        final_img.paste(new_img, (i * (num_sub_images * slice_width), 0))
    
    return final_img

# Usage
image_path = 'a.png' 
num_sub_images = 420
num_slices_per_image = 42

final_img = convert(image_path, num_sub_images, num_slices_per_image)   


final_img.save('final_image.png')
