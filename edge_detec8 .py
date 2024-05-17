import cv2
import numpy as np
from tensorflow.keras import  models

labels = ['good', 'bad', 'nochange', 'invalid']
img_size = 224
#loading the model
model = models.load_model('compile_rice_class_new03.model')

# Read the original image
img = cv2.imread('test_09.jpg')

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=255) # Canny Edge Detection
cv2.imshow('Canny Edge Detection', edges)
# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Group contours that are close to each other
grouped_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    found_group = False
    for idx, group in enumerate(grouped_contours):
        for c in group:
            
            if abs(x - c[0]) <= 70 and abs(y - c[1]) <= 80:
                grouped_contours[idx].append((x, y, w, h))
                found_group = True
                
                break
        if found_group:
            break
    if not found_group:
        grouped_contours.append([(x, y, w, h)])

# Combine rectangles that are very close to each other
combined_contours = []
for group in grouped_contours:
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
    for contour in group:
        x, y, w, h = contour
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    combined_contours.append((x_min, y_min, x_max, y_max))

# Remove rectangles that are fully contained within other rectangles
final_contours = []
for contour in combined_contours:
    x1, y1, x2, y2 = contour
    is_inside = False
    for c in combined_contours:
        if c is contour:
            continue
        x1_c, y1_c, x2_c, y2_c = c
        #if x1_c <= x1 and y1_c <= y1 and x2_c >= x2 and y2_c >= y2:
        if (x1_c <= x1 <= x2_c and y1_c <= y1 <= y2_c) or (x1_c <= x2 <= x2_c and y1_c <= y2 <= y2_c):
            is_inside = True
            break
    if not is_inside:
        final_contours.append(contour)
        #print(contour)

# Draw rectangles around final contours and save as separate jpg files
for i, contour in enumerate(final_contours):
    x1, y1, x2, y2 = contour
    # Add 25 pixels margin to the rectangle
    x1 -= 25
    y1 -= 25
    x2 += 25
    y2 += 25
    # Make sure the coordinates are within the image bounds
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    # Extract the region of interest
    roi = img[y1:y2, x1:x2]
    # Save the region of interest as a jpg file
    #i=145
    v_img = 'xx.jpg'
    cv2.imwrite(f'xx.jpg', roi)
    img_1 = cv2.imread(v_img)#open the image
    img_1= cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)#convert the color
    img_1 = cv2.resize(img_1, (img_size, img_size))#resize the image
    xx = np.array([img_1])/255 #normalize the data
    predictions = np.argmax(model.predict(xx), axis=-1)#classify the image
    predictions = predictions.reshape(1,-1)[0]
    if labels[predictions[0]] != 'invalid':
        #cv2.imwrite(f'{labels[predictions[0]]}_{i+1}.jpg', roi)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        caption = f'{labels[predictions[0]]}_{i+1}'
        (text_width, text_height) = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0, 0)[0]
        text_offset_x = 0
        text_offset_y = int(text_height-10)   
        cv2.rectangle(img, (x1, y1 ), (x2, y1), (255, 255, 255), -1)
        cv2.putText(img, caption, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 0, cv2.LINE_AA)
        


# Display image with rectangles around clusters of detected edges
#cv2.imshow('Edges with Rectangles', img)
cv2.imwrite(f'image_with_rects.jpg', img)
