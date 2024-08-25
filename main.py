import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import grayscale


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    # cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    # cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    if mor_op == 'dilate':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV Dilation Image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('Manual Dilation Image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
    elif mor_op == 'erode':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV Erosion Image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('Manual Erosion Image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual
        
    elif mor_op == 'opening':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.open(img, kernel)
        cv2.imshow('Manual Opening Image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual
        
    elif mor_op == 'closing':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV Closing Image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.close(img, kernel)
        cv2.imshow('Manual Closing Image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
        
    elif mor_op == 'hit_or_miss':
        kernel_hitmiss=np.array([[0,1,0],[1,-1,1],[0,1,0]])
        
        img_hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_hitmiss)
        cv2.imshow('OpenCV Hit or Miss Image', img_hit_or_miss)
        cv2.waitKey(wait_key_time)

        img_hit_or_miss_manual = binary.hitmiss(img, kernel_hitmiss)
        cv2.imshow('Manual Hit or Miss Image', img_hit_or_miss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hit_or_miss_manual
        
    elif mor_op == 'thinning':
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV Thinning Image', img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = binary.thin(img)
        cv2.imshow('Manual Thinning Image', img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual
        
    elif mor_op == 'extract_boundary':
        img_boundary = img-cv2.erode(img,kernel)
        cv2.imshow('OpenCV boundary extraction image', img_boundary)
        cv2.waitKey(wait_key_time)  

        img_boundary_manual = binary.extract_boundary(img,kernel)
        cv2.imshow('manual boundary extraction image', img_boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_manual
    
    elif mor_op == 'fillhole':
        #OpenCV
        img_floodfill = img.copy()
        cv2.floodFill(img_floodfill, None, (0,0), 255)
        img_floodfill_inv = cv2.bitwise_not(img_floodfill)
        img_hole_filling = img+img_floodfill_inv
        cv2.imshow('OpenCV hole filling image', img_hole_filling)
        cv2.waitKey(wait_key_time)

        #manual
        img_hole_filling_manual = binary.fill_hole(img,kernel)
        cv2.imshow('manual hole filling image',img_hole_filling_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hole_filling_manual

    elif mor_op == 'connectedcomponents':
        num, labels = cv2.connectedComponents(img, connectivity=8)
        img_labels_show = imshow_components(labels)
        cv2.imshow('OpenCV connected components image', img_labels_show)
        cv2.waitKey(wait_key_time)

        info, labels_manual = binary.extract_connected_components(img, kernel)
        img_labels_show_manual = imshow_components(labels_manual)

        for i in info:
            print('Number pixels of connected component {} : {}'.format(i, info[i]))
        cv2.imshow('manual connected components image', img_labels_show_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_labels_show_manual

    elif mor_op=='convexhull':
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = [cv2.convexHull(c) for c in contours]
        img_convex_hull_opencv = cv2.drawContours(img.copy(), hull, -1, (255, 0, 0), 2)
        cv2.imshow('OpenCV convex hull image', img_convex_hull_opencv)
        cv2.waitKey(wait_key_time)
        
        img_convex_hull_manual = binary.convex_hull(img)
        # contours_manual, _ = cv2.findContours(img_convex_hull_manual, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # img_convex_hull_manual_contour = cv2.cvtColor(img_convex_hull_manual, cv2.COLOR_GRAY2BGR)  # Chuyển sang BGR để vẽ màu
        # cv2.drawContours(img_convex_hull_manual_contour, contours_manual, -1, (255, 0, 0), 2)
        cv2.imshow('manual convex hull image', img_convex_hull_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_convex_hull_manual

    elif mor_op=='thicken':
        
        img_thickening_manual = binary.thicken(img)
        cv2.imshow('manual thickening image', img_thickening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thickening_manual
        
    elif mor_op == 'skeleton':
        img_skeleton_opencv = skeletonize(img)
        cv2.imshow('OpenCV skeleton image', img_skeleton_opencv)
        cv2.waitKey(wait_key_time)
        
        
        img_skeleton_manual = binary.skeletonize(img,kernel)
        cv2.imshow('manual skeleton image', img_skeleton_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_skeleton_manual

    elif mor_op == 'prun':
        # kernel = np.ones((3,3), np.uint8)
        # img_erosion = cv2.erode(img, kernel, iterations = 1)
        # img_dilation = cv2.dilate(img_erosion, kernel, iterations = 1)
        # img_pruning_opencv = cv2.absdiff(img, img_dilation)
        img_pruning_opencv = prun(img)
        cv2.imshow('OpenCV pruning image', img_pruning_opencv)
        cv2.waitKey(wait_key_time)
        
        
        #manual
        img_pruning_manual = binary.prun(img)
        cv2.imshow('manual pruning image', img_pruning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_pruning_manual
        
        
    # Grayscale image    
    
    elif mor_op == 'erode_gray':
        img_erosion = cv2.morphologyEx(img_gray,cv2.MORPH_ERODE,kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = grayscale.erode(img_gray, kernel)
        cv2.imshow('manual ersion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual

    elif mor_op=='dilate_gray':
        img_dilation = cv2.morphologyEx(img_gray,cv2.MORPH_DILATE,kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = grayscale.dilate(img_gray, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual
    
    elif mor_op=='open_gray':
        img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = grayscale.open(img_gray, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    elif mor_op=='close_gray':
        
        img_closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = grayscale.close(img_gray, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
    
    elif mor_op=='smooth_gray':
        img_opening =cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,kernel)
        img_smoothing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV smoothing image', img_smoothing)
        cv2.waitKey(wait_key_time)

        img_smoothing_manual = grayscale.smooth(img_gray,kernel)
        cv2.imshow('manual smoothing image', img_smoothing_manual)
        cv2.waitKey(wait_key_time)

        img_out=img_smoothing_manual
    
    elif mor_op=='gradient_gray':
        img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('OpenCV gradient image', img_gradient)
        cv2.waitKey(wait_key_time)

        img_gradient_manual = grayscale.gradient(img_gray,kernel)
        cv2.imshow('manual gradient image', img_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient_manual

    elif mor_op=='tophat_gray':
        img_top_hat= cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV top-hat image', img_top_hat)
        cv2.waitKey(wait_key_time)

        img_top_hat_manual = grayscale.top_hat(img_gray,kernel)
        cv2.imshow('manual top-hat image',img_top_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_top_hat_manual

    elif mor_op=='blackhat_gray':
        img_bottom_hat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV black-hat image', img_bottom_hat)
        cv2.waitKey(wait_key_time)
        
        img_bottom_hat_manual = grayscale.black_hat(img_gray,kernel)
        cv2.imshow('manual black-hat image',img_bottom_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_bottom_hat_manual
    
    elif mor_op=='granulometry_gray':
        # img_granulometry = granulometry_gray(img,2)
        # cv2.imshow('OpenCV granulometry image', img_granulometry)
        # # print(img_granulometry)
        # cv2.waitKey(wait_key_time)
        kernel = np.ones((8, 8), np.uint8)
        img_granulometry_manual = grayscale.granulometry(img_gray,kernel)
        cv2.imshow('manual granulometry image', img_granulometry_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_granulometry_manual
    
    if img_out is not None:
        cv2.imwrite(out_file, img_out)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def skeletonize(img):

    # Create an empty skeleton
    skel = np.zeros(img.shape, np.uint8)

    # Get a cross-shaped kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Erode the image
        eroded = cv2.erode(img, element)
        
        # Dilate the eroded image
        temp = cv2.dilate(eroded, element)
        
        # Subtract the dilated image from the original image
        temp = cv2.subtract(img, temp)
        
        # Or the result with the skeleton
        skel = cv2.bitwise_or(skel, temp)
        
        # Set the eroded image as the new input
        img = eroded.copy()

        # If there are no more white pixels, break
        if cv2.countNonZero(img) == 0:
            break
    

    return skel

def prun(img):
    set_8_kernel = [
        np.array([[0,-1,-1],[1,1,-1],[0,-1,-1]], dtype=np.int8),
        np.array([[0,1,0],[-1,1,-1],[-1,-1,-1]], dtype=np.int8),
        np.array([[-1,-1,0],[-1,1,1],[-1,-1,0]], dtype=np.int8),
        np.array([[-1,-1,-1],[-1,1,-1],[0,1,0]], dtype=np.int8),
        np.array([[1,-1,-1],[-1,1,-1],[-1,-1,-1]], dtype=np.int8),
        np.array([[-1,-1,1],[-1,1,-1],[-1,-1,-1]], dtype=np.int8),
        np.array([[-1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=np.int8),
        np.array([[-1,-1,-1],[-1,1,-1],[1,-1,-1]], dtype=np.int8)
    ]

    # Convert image to uint8 if it's not already
    img = np.uint8(img)

    tmp1 = img - cv2.morphologyEx(img, cv2.MORPH_HITMISS, set_8_kernel[0])
    tmp2 = tmp1 - cv2.morphologyEx(tmp1, cv2.MORPH_HITMISS, set_8_kernel[1])
    tmp3 = tmp2 - cv2.morphologyEx(tmp2, cv2.MORPH_HITMISS, set_8_kernel[2])
    tmp4 = tmp3 - cv2.morphologyEx(tmp3, cv2.MORPH_HITMISS, set_8_kernel[3])
    tmp5 = tmp4 - cv2.morphologyEx(tmp4, cv2.MORPH_HITMISS, set_8_kernel[4])
    tmp6 = tmp5 - cv2.morphologyEx(tmp5, cv2.MORPH_HITMISS, set_8_kernel[5])
    tmp7 = tmp6 - cv2.morphologyEx(tmp6, cv2.MORPH_HITMISS, set_8_kernel[6])
    tmp8 = tmp7 - cv2.morphologyEx(tmp7, cv2.MORPH_HITMISS, set_8_kernel[7])

    X1 = tmp8
    X1_before = X1
    X2 = np.zeros_like(img)
    for kernel in set_8_kernel:
        X1_after = cv2.morphologyEx(X1_before, cv2.MORPH_HITMISS, kernel)
        X2 += X1_after
        X1_before = X1_after

    H = np.ones((3,3), dtype=np.uint8)
    X3 = cv2.bitwise_and(cv2.dilate(X2, H), img)
    return X1 + X3

def granulometry_gray(image, max_kernel_size):
    granulometry_results = []
    for size in range(1, max_kernel_size + 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        sum_pixels = np.sum(opened_image)
        granulometry_results.append(sum_pixels)
    granulometry_results = np.array(granulometry_results, dtype=np.uint8)
    return granulometry_results

def operator_reconstruction(in_file,marker_file,iteration, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    # cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)
  
    kernel = np.ones((5, 5), np.uint8)

    img_out = None

    # Marker
    marker_origin = cv2.imread(marker_file)


    marker_gray  = cv2.imread(marker_file,0)  
    cv2.imshow('gray marker image', marker_gray)
    cv2.waitKey(wait_key_time)

    (thresh_re, marker) = cv2.threshold(marker_gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary marker image', marker)
    cv2.waitKey(wait_key_time)

    if mor_op == 'reconstruct_by_dilation':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_reconstruct_by_dialtion = binary.reconstruct_by_dilation(marker_resized, kernel,img)
        cv2.imshow('manual reconstruction by dilation', img_reconstruct_by_dialtion)
        cv2.waitKey(wait_key_time)

        img_out = img_reconstruct_by_dialtion

    elif mor_op == 'reconstruct_by_erosion':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_reconstruct_by_erosion = binary.reconstruct_by_erosion(marker_resized,kernel,img)
        cv2.imshow('manual reconstruction by erosion', img_reconstruct_by_erosion)
        cv2.waitKey(wait_key_time)
        
        img_out = img_reconstruct_by_erosion
    
    elif mor_op=='open_by_reconstruction':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_open_by_reconstruction = binary.open_by_reconstruction(marker_resized,kernel,img, iteration)
        cv2.imshow('manual opening by reconstruction image',img_open_by_reconstruction)
        cv2.waitKey(wait_key_time)

        img_out = img_open_by_reconstruction
    
    elif mor_op=='close_by_reconstruction':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_close_by_reconstruction = binary.close_by_reconstruction(marker_resized, kernel, img, iteration)
        cv2.imshow('manual closing by reconstruction',img_close_by_reconstruction)
        cv2.waitKey(wait_key_time)

        img_out = img_close_by_reconstruction

    elif mor_op =='reconstruct_by_dilation_gray':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker_gray, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_reconstruct_by_dialtion_gray = grayscale.reconstruct_by_dilation(marker_resized, kernel, img_gray)
        cv2.imshow('manual reconstruction by dilation',img_reconstruct_by_dialtion_gray)
        cv2.waitKey(wait_key_time)

        img_out = img_reconstruct_by_dialtion_gray
    
    elif mor_op=='reconstruct_by_erosion_gray':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker_gray, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_reconstruct_by_erosion_gray = grayscale.reconstruct_by_erosion(marker_resized, kernel, img_gray)
        cv2.imshow('manual reconstruction by dilation',img_reconstruct_by_erosion_gray)
        cv2.waitKey(wait_key_time)

        img_out = img_reconstruct_by_erosion_gray

    elif mor_op=='open_by_reconstruction_gray':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker_gray, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_open_by_reconstruction_gray = grayscale.open_by_reconstruction(marker_resized,kernel,img_gray,iteration)
        cv2.imshow('manual opening by reconstruction',img_open_by_reconstruction_gray)
        
        cv2.waitKey(wait_key_time)

        img_out = img_open_by_reconstruction_gray

    elif mor_op=='close_by_reconstruction_gray':
        if marker.shape != img.shape:
            # Resize 'marker' to match 'img' shape
            marker_resized = cv2.resize(marker_gray, (img.shape[1], img.shape[0]))
        else:
            marker_resized = marker
        img_open_by_reconstruction_gray = grayscale.close_by_reconstruction(marker_resized,kernel,img_gray,iteration)
        cv2.imshow('manual opening by reconstruction',img_open_by_reconstruction_gray)
        cv2.waitKey(wait_key_time)

        img_out = img_open_by_reconstruction_gray


    if img_out is not None:
        cv2.imwrite(out_file, img_out)

def main(argv):
    input_file = ''
    marker_file = ''
    iteration=0
    output_file = ''
    mor_op = ''
    wait_key_time = 0
    argc = len(sys.argv)

    description = 'main.py -i <input_file> -m <marker>(for reconstruct) -k <iteration>(for open close reconstruct) -o <output_file> -p <mor_operator> -t <wait_key_time>'
    try:
        opts = ''
        if(argc==9):
            opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
        elif(argc==11):
            opts, args = getopt.getopt(argv, "hi:m:o:p:t:", ["in_file=", "marker_file=","out_file=", "mor_operator=", "wait_key_time="])
        elif(argc==13):
            opts, args = getopt.getopt(argv, "hi:m:k:o:p:t:", ["in_file=", "marker_file=","iteration=","out_file=", "mor_operator=", "wait_key_time="])
        else:
            print(description)
    except getopt.GetoptError:
        print(description)
        sys.exit(2)
    if(argc==9):
        for opt, arg in opts:
            if opt == '-h':
                print(description)
                sys.exit()
            elif opt in ("-i", "--in_file"):
                input_file = arg
            elif opt in ("-o", "--out_file"):
                output_file = arg
            elif opt in ("-p", "--mor_operator"):
                mor_op = arg
            elif opt in ("-t", "--wait_key_time"):
                wait_key_time = int(arg)

        print('Input file is ', input_file)
        print('Output file is ', output_file)
        print('Morphological operator is ', mor_op)
        print('Wait key time is ', wait_key_time)

        operator(input_file, output_file, mor_op, wait_key_time)
        cv2.waitKey(wait_key_time)

    elif(argc==11):
        for opt, arg in opts:
            if opt == '-h':
                print(description)
                sys.exit()
            elif opt in ("-i", "--in_file"):
                input_file = arg
            elif opt in ("-m", "--marker_file"):
                marker_file = arg
            elif opt in ("-o", "--out_file"):
                output_file = arg
            elif opt in ("-p", "--mor_operator"):
                mor_op = arg
            elif opt in ("-t", "--wait_key_time"):
                wait_key_time = int(arg)

        print('Input file is ', input_file)
        print('Marker file is ',marker_file)
        print('Output file is ', output_file)
        print('Morphological operator is ', mor_op)
        print('Wait key time is ', wait_key_time)

        operator_reconstruction(input_file, marker_file,0, output_file, mor_op, wait_key_time)
        cv2.waitKey(wait_key_time)
    elif(argc==13):
        iteration = 0
        for opt, arg in opts:
            if opt == '-h':
                print(description)
                sys.exit()
            elif opt in ("-i", "--in_file"):
                input_file = arg
            elif opt in ("-m", "--marker_file"):
                marker_file = arg
            elif opt in("-k", "--iteration"):
                iteration = int(arg)
            elif opt in ("-o", "--out_file"):
                output_file = arg
            elif opt in ("-p", "--mor_operator"):
                mor_op = arg
            elif opt in ("-t", "--wait_key_time"):
                wait_key_time = int(arg)

        print('Input file is ', input_file)
        print('Marker file is ',marker_file)
        print('Iteration is ',iteration)
        print('Output file is ', output_file)
        print('Morphological operator is ', mor_op)
        print('Wait key time is ', wait_key_time)

        operator_reconstruction(input_file, marker_file,iteration, output_file, mor_op, wait_key_time)
        cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
