import os
import cv2
import geojson
import tifffile
import shapely.wkt
import numpy as np


# Main function to create a mask from json file
def create_mask(json_file):
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    
    with open(json_file) as file:
        gj = geojson.load(file) 
        
    # loop to go through all the Annotations in a json file
    for k in range(len(gj['features']['xy'])):
        
        # To get the polygon structure of the coordinates for each annotation
        g1 = shapely.wkt.loads(gj['features']['xy'][k]['wkt'])
        
        # Convert the polygon into a geojson feature structure 
        g2 = geojson.Feature(geometry=g1, properties={})
        
        # Convert the list of all points in the annotation into an numpy array
        pts = np.array(g2.geometry["coordinates"]).astype(np.int32)

        # To get the annotations for either pre or post disaster labels
        try:
            type = gj['features']['xy'][k]['properties']['subtype']
        except:
            type = 'building'

        if type == 'minor-damage':
            mask = cv2.fillPoly(mask, pts, color=2) 
        elif type == 'major-damage':
            mask = cv2.fillPoly(mask, pts, color=3)
        elif type == 'destroyed':
            mask = cv2.fillPoly(mask, pts, color=4)
        else:
            mask = cv2.fillPoly(mask, pts, color=1)
            
    return mask 

            
def create_visible_mask(path_json):

    mask = create_mask(path_json)

    # to colorize the mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    mask[np.all(mask == (1,1,1), axis=-1)] = (51,255,255)
    mask[np.all(mask == (2,2,2), axis=-1)] = (255,255,0)
    
    mask[np.all(mask == (3,3,3), axis=-1)] = (255,128,0)
    
    mask[np.all(mask == (4,4,4), axis=-1)] = (255,0,0) 

    return mask 
    
    # return cv2.resize(mask, (512, 512), interpolation = cv2.INTER_LINEAR)


# cut_mask & cut image for Model Visualization  

def cut_mask(path):
    mask = create_visible_mask(path)
    return mask[:512, :512, :], mask[:512, 512:1024, :], mask[512:1024, :512, :], mask[512:1024, 512:1024, :]

def cut_image(path):
    tif = tifffile.imread(path).astype('uint8')
    tif = tif/255
    tif = np.expand_dims(tif, axis=0)
    return tif[:, :512, :512, :], tif[:, :512, 512:1024, :], tif[:, 512:1024, :512, :], tif[:, 512:1024, 512:1024, :]


# cut_msk & cut img for Model Evaluation 

def cut_msk(path):
    mask = create_mask(path)
    return mask[:512, :512], mask[:512, 512:1024], mask[512:1024, :512], mask[512:1024, 512:1024]


def cut_img(path):
    tif = tifffile.imread(path).astype('uint8')
    return tif[:512, :512, :], tif[:512, 512:1024, :], tif[512:1024, :512, :], tif [ 512:1024, 512:1024, :]



def overlay_transparent(bg_img, img_to_overlay_t):
    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    mask = cv2.medianBlur(a,5)

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(bg_img.copy(),bg_img.copy(),mask = cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    # Update the original image with our new ROI
    bg_img = cv2.add(img1_bg, img2_fg)

    return bg_img
    

def black_to_transparent(img):
        # Convert image to image gray
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying thresholding technique
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

        # Using cv2.split() to split channels 
        # of coloured image
        b, g, r = cv2.split(img)

        # Making list of Red, Green, Blue
        # Channels and alpha
        rgba = [b, g, r, alpha]

        # Using cv2.merge() to merge rgba
        # into a coloured/multi-channeled image
        return cv2.merge(rgba, 4)
    