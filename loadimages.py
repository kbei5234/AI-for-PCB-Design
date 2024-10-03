import pandas as pd
import numpy as np
import os
from PIL import Image
import cairosvg


# the path to the image folder and labels.csv file
image_folder = '/Users/yzy/Documents/goodchip/generated_images' # change the path based off your current path
label_file = '/Users/yzy/Documents/goodchip/labels.csv' # change the path based off your current path


# Load the CSV file using pandas
labels_df = pd.read_csv(label_file)
labels_df = labels_df.reset_index()


# image size for resizing
image_size = (400, 400)


image_list = []
label_list = []


for index, row in labels_df.iterrows():
   file_name = row.iloc[1]  # the filename from the CSV
   label = row.iloc[2]  # the label from the CSV


   print(f"Reading in file: {file_name}")
   print(f"Label: {label}")


   # Complete path to SVG file
   svg_path = os.path.join(image_folder, file_name)
   png_path = svg_path.replace('.svg', '.png')


   # we are changing from a SVG to PNG file
   if os.path.exists(svg_path):
       try:
           cairosvg.svg2png(url=svg_path, write_to=png_path)
       except Exception as e:
           print(f"Exist an error converting {file_name}: {e}")
           continue 


       # Check if the PNG file was created and process it
       if os.path.exists(png_path):
           try:
               img = Image.open(png_path)
               img = img.resize(image_size) 
               img_array = np.array(img)
               image_list.append(img_array)
               label_list.append(label)
           except Exception as e:
               print(f"Error processing {png_path}: {e}")
       else:
           print(f"PNG file {png_path} not found.")
   else:
       print(f"SVG file {svg_path} not found.")


# change the lists to a numpy array
X = np.array(image_list)  # Images
y = np.array(label_list)  # Labels


# Check if there is an equal amount of images and labels
if len(X) != len(y):
   print(f"Warning: Number of images ({len(X)}) is not equal to the number of labels ({len(y)}).")


print(f"Read in {len(X)} images and {len(y)} labels.")


# Save data to files
np.save('X.npy', X) 
np.save('y.npy', y)
print(y)
