#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
import copy
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def get_channel_color(img,chnl=None):
    """
    This will return the image with only one of the specified color channels. 
    If no color or worng color is selected, function will raise ValueError 
    """
    temp_img = copy.copy(img)
    if chnl == 'red':
        #remove green channel
        temp_img[:,:,1] = 0

        #remove blue channel
        temp_img[:,:,2] = 0
    elif chnl == 'green':
        #remove red channel
        temp_img[:,:,0] = 0

        #remove blue channel
        temp_img[:,:,2] = 0
    elif chnl == 'blue':
        #remove red channel
        temp_img[:,:,0] = 0

        #remove green channel
        temp_img[:,:,1] = 0
    else:
        raise ValueError("Channel color must be red,green or blue i.e. chnl='red', chnl='green', or chnl='blue'") 
        
    return temp_img
    
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def generate_line_points(x1,y1,x2,y2,m=None,b=None,stepLen=3):
    xPts = np.array([])
    yPts = np.array([])
    if m == None or b == None:
        #calculate m and b
        m = ((y2-y1)/(x2-x1)) if (x2-x1) != 0 else np.infty
        b = y1 - (m*x1)
    
    if abs(x2-x1) > stepLen:
        numPts = np.int32(np.floor(2 + (abs(x2-x1)/stepLen)))
        if x1 < x2:
            for x in range(x1,x2,numPts):
                xPts = np.append(xPts,x) 
                yPts = np.append(yPts,np.poly1d([m,b])(x))
        else:
            for x in range(x2,x1,numPts):
                xPts = np.append(xPts,x) 
                yPts = np.append(yPts,np.poly1d([m,b])(x))
    else:
        xPts = np.append(xPts,[x1,x2]) 
        yPts = np.append(yPts,[y1,y2])
    
    return np.int32(xPts),np.int32(yPts)
        
def draw_lines(img, lines, color=[255, 0, 0]):
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1,y1), (x2,y2), [0, 0, 255], thickness=2)
            m = ((y2-y1)/(x2-x1)) if (x2-x1) != 0 else np.infty
            if abs(math.atan(m)) >= np.pi*20/180 and abs(math.atan(m)) <= np.pi*80/180:
                cv2.line(img, (x1,y1), (x2,y2), color, thickness=5)

def draw_polylines(img, lines, color=[255, 0, 0], thickness=15,xLimit=[100,920],yLimit=[340,540], degree=1):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    l1X = np.array([])
    l1Y = np.array([])
    
    l2X = np.array([])
    l2Y = np.array([])
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            # Filter out lines with lines with slope magnitude < 30 degrees
            m = ((y2-y1)/(x2-x1)) if (x2-x1) != 0 else np.infty
            if abs(math.atan(m)) <= np.pi*20/180 or abs(math.atan(m)) >= np.pi*80/180:
                continue
            else:
                # Build 4 numpy array. 2 array per line. I will build line of best fit for each side
                #lXPts,lYPts = generate_line_points(x1,y1,x2,y2)
                lXPts = [x1,x2]
                lYPts = [y1,y2]
                if m > 0:
                    l1X = np.append(l1X,lXPts)
                    l1Y = np.append(l1Y,lYPts)
                else:
                    l2X = np.append(l2X,lXPts)
                    l2Y = np.append(l2Y,lYPts)
    
    
    if np.shape(l1X)[0] != 0:
        # Calculate polyfit and store as poly1d equation              
        l1P = np.poly1d(np.polyfit(l1X,l1Y,degree))
        
        # Generate polyline points for polyline draw
        l1Pts = np.array([])
        
        for x in range(xLimit[0],xLimit[1]):
            y = l1P(x)
            if y >= yLimit[0] and y <= yLimit[1]:
                l1Pts = np.append(l1Pts,[x,y])
        
        # Reshape into point structure
        l1Pts = np.reshape(l1Pts,(-1,2))
        
        cv2.polylines(img, np.int32([l1Pts]), False, color, thickness)   
    
    if np.shape(l2X)[0] != 0:
        l2P = np.poly1d(np.polyfit(l2X,l2Y,degree))
        
        l2Pts = np.array([])
        
        for tx2 in range(xLimit[0],xLimit[1]):
            ty2 = l2P(tx2)
            if ty2 >= yLimit[0] and ty2 <= yLimit[1]:
                l2Pts = np.append(l2Pts,[tx2,ty2])
        
        l2Pts = np.reshape(l2Pts,(-1,2)) 
        
        cv2.polylines(img, np.int32([l2Pts]), False, color, thickness)
    
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap,xLimit=[100,920],yLimit=[340,540],degree=1):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img,lines)
    draw_polylines(line_img, lines,xLimit=xLimit,yLimit=yLimit,degree=degree)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
#Convert png screenshot to jpg of aproximate equal size as the other pictures in test_images
#temp_img = cv2.imread('test_images/challengeYellowLeftCurve1.png')
#cv2.imwrite('test_images/challengeYellowLeftCurve1.jpg',temp_img,[cv2.IMWRITE_JPEG_QUALITY,50])

#Load challenge images for experimentation and learning
cYRCurve1 = mpimg.imread('test_images/challengeYellowRightCurve1.jpg')
cYRCurve2 = mpimg.imread('test_images/challengeYellowRightCurve2.jpg')
print("Original")
plt.imshow(cYRCurve2)
plt.imshow(grayscale(cYRCurve2),cmap='gray')
plt.imshow(get_channel_color(cYRCurve2,'red'))
plt.imshow(gaussian_blur(get_channel_color(cYRCurve2,'green'),5))
plt.imshow(get_channel_color(cYRCurve2,'blue'))
plt.imshow(grayscale(get_channel_color(cYRCurve2,'blue')),cmap="gray")
def simple_canny_pipeline(img,threshLow=50,gKernel=0):
    # Force 1:2 ratio for low:high threshold
    threshHigh = 2*threshLow
    imgGry = grayscale(img)
    if gKernel == 0:
        blurImg = imgGry
    else:
        blurImg = gaussian_blur(imgGry,gKernel)  
    return canny(blurImg,threshLow,threshHigh)

def rgb_Canny(img, threshLow=50, gKernel=0):
    # Force 1:2 ratio for low:high threshold
    threshHigh = 2*threshLow
    # 1) Separate image to R,G,B Channels
    rImg = get_channel_color(img,chnl='red')
    gImg = get_channel_color(img,chnl='green')
    bImg = get_channel_color(img,chnl='blue')
    
    # 2) Conver each of the channels to greyscale
    rImgGry = grayscale(rImg)
    gImgGry = grayscale(gImg)
    bImgGry = grayscale(bImg)
    
    # 3) Blur each channel to remove some noise
    rBlurImg = gaussian_blur(rImgGry,gKernel)
    gBlurImg = gaussian_blur(gImgGry,gKernel)
    bBlurImg = gaussian_blur(bImgGry,gKernel)
    
    # 4) Calculated adjusted threshold for individual channels, 
    #    then apply Canny using adjusted threshold for each greyscale image
    rThreshLow =  np.uint8(np.floor(0.299*threshLow))
    rThreshHigh = np.uint8(np.floor(0.299*threshHigh))
    
    gThreshLow = np.uint8(np.floor(0.587*threshLow))
    gThreshHigh = np.uint8(np.floor(0.587*threshHigh))
    
    bThreshLow = np.uint8(np.floor(0.114*threshLow))
    bThreshHigh = np.uint8(np.floor(0.114*threshHigh))
    
    rCny = canny(rBlurImg,rThreshLow,rThreshHigh)
    gCny = canny(gBlurImg,gThreshLow,gThreshHigh)
    bCny = canny(bBlurImg,bThreshLow,bThreshHigh)
    
    # Return the OR of the r,g,b canny filter
    return np.uint8(np.logical_or(np.logical_or(rCny,gCny),bCny)*255)
c2SCImg = simple_canny_pipeline(cYRCurve2,80,9)
plt.imshow(c2SCImg,cmap='gray')
c2CImg = rgb_Canny(cYRCurve2,80,9)
plt.imshow(c2CImg,cmap='gray')
# Mask canny image before hough_lines
imshape = np.shape(c2CImg)
region = np.array([[(0.13*imshape[1],imshape[0]),(0.46*imshape[1],0.58*imshape[0]), (0.545*imshape[1],0.58*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
mskImg = region_of_interest(c2CImg, [region])
#mskImg = region_of_interest(c2CnyImg, [np.array([ [100,540], [370,340], [570,340], [920,540]])]) 
    
plt.imshow(mskImg,cmap="gray")
hlImg = hough_lines(mskImg,1,np.pi/360,8,3,2,xLimit=[int(0.13*imshape[1]),int(0.958*imshape[1])],yLimit=[int(0.61*imshape[0]),imshape[0]],degree=1)
plt.imshow(hlImg)
plt.imshow(weighted_img(cYRCurve2,hlImg))
def lane_line_pipeline(img,threshLow=50,gKernel=3,hThresh=10,minLineLen=2,maxLineGap=2,degree=1):
    # 1) Apply RGB Canny
    #cnyImg = simple_canny_pipeline(img,threshLow,threshHigh,gKernel)
    cnyImg = rgb_Canny(img,threshLow,gKernel)
    
    # 2) Mask image to select region of interest
    imshape = np.shape(cnyImg)
    region = np.array([[(0.14*imshape[1],imshape[0]),(0.46*imshape[1],0.6*imshape[0]), (0.545*imshape[1],0.6*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    mskImg = region_of_interest(cnyImg, [region])
    
    # 3) Apply Houg Line with line of best fit draw lines
    hlImg = hough_lines(mskImg,1,np.pi/360,hThresh,minLineLen,maxLineGap,xLimit=[int(0.13*imshape[1]),int(0.958*imshape[1])],yLimit=[int(0.61*imshape[0]),imshape[0]],degree=degree)
    
    # 4) Add a transparent line obtain from Houg Line to original image and return
    return weighted_img(img,hlImg)
import os
os.listdir("test_images/")
folder = "test_images/"
files = os.listdir(folder)
for imgName in files:
    fileName = folder+imgName
    img = mpimg.imread(fileName)
    llImg = lane_line_pipeline(img,threshLow=80,gKernel=9,hThresh=25,minLineLen=8,maxLineGap=5,degree=1)
    #conver from RGB to BGR for imwrite
    wImg = cv2.cvtColor(llImg,cv2.COLOR_RGB2BGR)
    cv2.imwrite(fileName[:-4]+"_WithLine"+fileName[-4:],wImg)    
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    
    # you should return the final output (image where lines are drawn on lanes)

    return lane_line_pipeline(image,threshLow=80,gKernel=9,hThresh=25,minLineLen=8,maxLineGap=5,degree=1)
    #return lane_line_pipeline(image,threshLow=80,gKernel=7,hThresh=15,minLineLen=3,maxLineGap=10,degree=1)
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
