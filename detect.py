#  For "sunflowers.jpg" ---->  set hyper-parameters as following:

#  [line 28]      Distance defined in Non-max_suppression() = sqrt(2)*maximum radius of the blobs identified 
#  [line 55 & 56] Size of sliced matrix to check argmax  [7x7] 
#  [line 61]      Threshold for blob detection = 0.009
#  [line 213]     sigma_initial = 1 
#  [line 214]     k = sqrt(2) 
#  [line 215]     Threshold for overlap checking = 0.35
#  [line 216]     Size of octave = 5
#  [line 217]     scale_for_sigma_to_kernel_size = 6 

import math
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pylab import *
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial


def non_max_suppression(blobs_array, overlap_threshold):
    print("\nWorking on Non-Maximum Suppression...Please wait !!")
    blobs_array_after_NMS = []
    max_radius_of_blob = blobs_array[:, -1].max()
    #distance = 2*max_radius_of_blob
    #print("\nDistance to make pair = ",distance)
    distance = sqrt(2) * max_radius_of_blob
    tree = spatial.cKDTree(blobs_array [:, :-1])
    #Make pair of keypoints whose distance from each other is at max equal to 'distance' 
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if overlap_ratio(blob1, blob2) > overlap_threshold:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
    for blobs in blobs_array:
        #if blobs[-1]>0:
        if blobs[-1]>1:
            blobs_array_after_NMS.append(blobs)
    blobs_array_after_NMS = np.asarray(blobs_array_after_NMS)
    return blobs_array_after_NMS
   
def detect_blob(dog_images, sigma, k, threshold, sz):
    #to store co ordinates
    co_ordinates = []
    (h,w) = dog_images[0].shape
    a = 0
    #Set the ranges of i and j accordingly to make the shape of sliced matrix equal to 3x3, 5x5, 7x7 or anything else
    for i in range(sz//2,h-(sz//2)):
        for j in range(sz//2,w-(sz//2)):
            #10*5*5 slice
            sliced_matrix_in_octave = dog_images[:,i-(sz//2):i+((sz//2)+1),j-(sz//2):j+((sz//2)+1)]
            max_value_in_sliced_matrix = np.amax(sliced_matrix_in_octave) 
            #print ("Max value = ",max_value_in_sliced_matrix)
            #Set Threshold Value to detect any peak in the image and ignore the rest
            if max_value_in_sliced_matrix >= threshold: 
                z,x,y = np.unravel_index(sliced_matrix_in_octave.argmax(),sliced_matrix_in_octave.shape)
                #finding co-ordinates - x, y, radius
                co_ordinates.append((i+x-1,j+y-1,k**(z)*sigma)) 
            else:
                #print("Blob not found")
                a=1
    return co_ordinates

def get_kernal(size,sigma):
    ker = cv.getGaussianKernel(int(size), sigma)
    ker = np.outer(ker,ker)
    return ker

def get_kernal_octave(sigma_initial,k,size_of_octave,scale_for_sigma_to_kernel_size):
    # we create an empty array to store various sigma values
    gaussian_kernal_with_diff_sigma = []
    sigma_values = []
    for i in range (size_of_octave):
        sigma_next = (k**i)*sigma_initial
        sigma_values.append(sigma_next)
    for kernal_index_in_octave in range(size_of_octave):
        kernel_size = np.ceil(scale_for_sigma_to_kernel_size*sigma_values[kernal_index_in_octave])
        kernal = get_kernal(kernel_size,sigma_values[kernal_index_in_octave])
        gaussian_kernal_with_diff_sigma.append(kernal)  
    return gaussian_kernal_with_diff_sigma

#Function that returns the overlap ration between two blobs
def overlap_ratio(blob1,blob2):
    radius_blob1 = blob1[-1]
    radius_blob2 = blob2[-1]
    distance_between_blobs = sqrt(((blob2[0]-blob1[0])**2)+((blob2[1]-blob1[1])**2))
    if (distance_between_blobs > (radius_blob1 + radius_blob2)):
        return 0
    elif distance_between_blobs <= abs(radius_blob1 - radius_blob2):
        return 1
    else:
        d1 = np.abs((((radius_blob1)**2)-((radius_blob2)**2)+((distance_between_blobs)**2))/(2*distance_between_blobs))
        d2 = np.abs((((radius_blob2)**2)-((radius_blob1)**2)+((distance_between_blobs)**2))/(2*distance_between_blobs))
        
        theta_1 = 2* (math.acos ((round((d1/radius_blob1),4))))
        theta_2 = 2* (math.acos ((round((d2/radius_blob2),4))))
                                  
        base_of_triangle = 2*radius_blob1*sin(theta_1/2)
                                  
        Area_S1 = (theta_1/(2*np.pi))*(np.pi*(radius_blob1**2))-((1/2)*(base_of_triangle)*(d1))
        Area_S2 = (theta_2/(2*np.pi))*(np.pi*(radius_blob2**2))-((1/2)*(base_of_triangle)*(d2))
        Area_of_overlap = Area_S1 + Area_S2
                                  
        ratio_of_overlap = Area_of_overlap/(np.pi*((min(radius_blob1,radius_blob2))**2))
        return ratio_of_overlap

#Display images
def display(convolved_images):  
    for h in range(len(convolved_images)):
        plt.imshow(convolved_images[h],cmap='gray')
        plt.show()

#DFT2 function defination
def DFT2(f):
    sz1 = f.shape
    im_eq = f
    img = im_eq
    f_1 = np.zeros((sz1[0],sz1[1]), dtype="complex_")
    f_2 = np.zeros((sz1[0],sz1[1]), dtype="complex_")
    for i in range(sz1[0]):
        f_1[i,:] = np.fft.fft(img[i,:])
    for j in range(sz1[1]):    
        f_2[:,j] = np.fft.fft(f_1[:,j])
    F = f_2
    return F

#IDFT2 function defination  
def IDFT2(product):    
    sz_F = product.shape
    #Again re-shift the origin to top left corner of the image
    corner_shifted_product = np.fft.ifftshift(product)
    f_3 = np.zeros((sz_F[0],sz_F[1]), dtype="complex_")
    f_4 = np.zeros((sz_F[0],sz_F[1]), dtype="complex_")
    #Finding 2D inverse of F using 1D inbuilt function
    for i in range(sz_F[1]):
        f_3[:,i] = np.fft.ifft(corner_shifted_product[:,i])
    for j in range(sz_F[0]):    
        f_4[j,:] = np.fft.ifft(f_3[j,:])
    g = f_4
    return g

def conv2(f,kernal):
      
    #Calling the DFT2 function for input image and passing normalized image
    F_image = DFT2(f/255)
    #Spectrum Shifted to centre
    centre_shifted_dft2_img = np.fft.fftshift(F_image)

    kernal_original = kernal

    #Padding done on kernal to make size of kernaL equal to size of input image
    kernal_resized_to_input_image = np.zeros((img_original.shape[0],img_original.shape[1]))
    
    rows_input_image = f.shape[0]
    cols_input_image = f.shape[1]
    
    row_lower_limit = int(np.floor((rows_input_image-kernal.shape[0])/2))
    row_upper_limit = row_lower_limit+kernal.shape[0]
    
    col_lower_limit = int(np.floor((cols_input_image-kernal.shape[1])/2))
    col_upper_limit = col_lower_limit + kernal.shape[1]
    
    kernal_resized_to_input_image[row_lower_limit:row_upper_limit,col_lower_limit: col_upper_limit] = kernal

    #Calling the DFT2 function for kernal 
    F_kernal = DFT2(kernal_resized_to_input_image)
    #Spectrum shifted to centre
    centre_shifted_dft2_kernal = np.fft.fftshift(F_kernal)

    #Convolution in frequency domain
    Product_img = F_image*F_kernal

    #Inverse fourier transform of product to get convoluted image
    inverse_fft_product = IDFT2(Product_img)

    #Centre shifted prodcut of convolution in frequency domain
    centre_shift_inverse_transpose_product = np.fft.fftshift(inverse_fft_product)
    return np.abs(centre_shift_inverse_transpose_product.real)

def D_o_G(gaussian_smoothened_pyramid):
    difference_of_gaussian_images = []
    length_of_pyramid = len(gaussian_smoothened_pyramid)
    for index_in_dog in range(length_of_pyramid-1):
        difference_between_2_layers = np.abs(np.subtract(gaussian_smoothened_pyramid[index_in_dog],gaussian_smoothened_pyramid[index_in_dog+1]))
        difference_of_gaussian_images.append(difference_between_2_layers)
    return difference_of_gaussian_images

if __name__ == "__main__":
    
    selected_file = int(input("Please type the serial number of the input image to run the Blob detection on : \n" +
                "1. Butterfly\n" +
                "2. Einstein\n" +
                "3. Fishes\n" +
                "4. Sunflowers\n" +
                "5. Cricket\n" +
                "6. Football\n" +
                "7. Dog\n" +
                "8. Cat\n"))

    filenames = {
        1:'butterfly.jpg',
        2:'einstein.jpg',
        3:'fishes.jpg',
        4:'sunflowers.jpg',
        5:'cricket.jpg',
        6:'football.jpg',
        7:'dog.jpg',
        8:'cat.jpg'
    }

    f = cv.imread(filenames.get(selected_file,"Invalid input.\n"))
    f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)

    #Histogram Equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    f = clahe.apply(f)
    sz = f.shape
    g = np.zeros((sz[0],sz[1]), dtype="complex_")
    img_original = f
    
    sigma_initial = 1
    k = math.sqrt(2)
    scale_for_sigma_to_kernel_size = 6

    #Hyper-parameter array = [blob_detection_threshold, overlap_threshold, size_of_octave, size_of_matrix]
    hyper_parameters = []
    parameters = {
        1:[0.01, 0.3, 10, 9],
        2:[0.053, 0.3, 10, 11],
        3:[0.009, 0.35, 5, 7],
        4:[0.0012, 0.2, 10, 7],
        5:[0.04, 0.2, 10, 9],
        6:[0.008, 0.2, 10, 7],
        7:[0.03, 0.2, 5, 7],
        8:[0.05, 0.5, 10, 5]
    }
    hyper_parameters = parameters.get(selected_file,"Invalid input.\n")

    blob_detection_threshold = hyper_parameters[0]
    overlap_threshold = hyper_parameters[1]
    size_of_octave = hyper_parameters[2]
    sz = hyper_parameters[3]

    start_time = time.process_time()
    kernal_octave = get_kernal_octave(sigma_initial,k,size_of_octave,scale_for_sigma_to_kernel_size)
    end_time = time.process_time()
    print(f"Time Elapsed [get_kernal_octave()] = ", (end_time - start_time))

    #Create an empty array to store gaussian smoothened images for different sigma values
    convolved_image_in_octave = []
    
    #Time calculation to check process time : COnvolution done in 0.1 sec in frequency domain
    start_time = time.process_time()
    for octave_index in range(size_of_octave):
        convolved_image = conv2(f.copy(),kernal_octave[octave_index])
        convolved_image_in_octave.append(convolved_image)
    end_time = time.process_time()
    print(f"Time Elapsed in making octave of convolved image = ", (end_time - start_time))
    #Calculate Difference of Gaussian 
    dog_images = []
    start_time = time.process_time()
    dog_images = D_o_G(convolved_image_in_octave)
    dog_images = np.asarray(dog_images)
    end_time = time.process_time()
    print(f"Time Elapsed [D_o_G()] = ", (end_time - start_time))
    
    #print("Displaying Gaussian pyramid with different sigma")
    #display(convolved_image_in_octave)

    #print("Displaying Difference of Gaussian pyramid with different sigma")
    #display(dog_images)

    start_time = time.process_time()
    co_ordinates = list(set(detect_blob(dog_images, sigma_initial, k, blob_detection_threshold, sz)))
    co_ordinates = np.array(co_ordinates)
    print("\nNumber of Blobs detected BEFORE Non Max Suppression", len(co_ordinates))
    end_time = time.process_time()
    print(f"Time Elapsed [detect_blob()] = ", (end_time - start_time))
    start_time = time.process_time()
    co_ordinates = non_max_suppression(co_ordinates,overlap_threshold)
    print("\nNumber of Blobs detected AFTER Non Max Suppression", len(co_ordinates))
    end_time = time.process_time()
    print(f"\nTime Elapsed [non_maximum_suppression()] = ", (end_time - start_time))
    fig, ax = plt.subplots()
    nh,nw = f.shape
    count = 0
    ax.imshow(f, interpolation='nearest',cmap="gray")
    print("\nPlotting blobs on the image...Please wait !!")
    stAXart_time = time.process_time()
    for blob in co_ordinates:
        y,x,r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=0.5, fill=False)
        #c = plt.Circle((x, y), r*3, color='red', linewidth=0.5, fill=False)
        ax.add_patch(c)
    ax.plot()  
    end_time = time.process_time()
    print(f"Time Elapsed [plotting keypoints] = ", (end_time - start_time))
    plt.show()