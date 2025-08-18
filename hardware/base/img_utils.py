import cv2

def combine_image(img1:cv2.Mat, img2:cv2.Mat):
    # 找到两张图像中较小的尺寸
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    
    # 将两张图像都resize到较小的尺寸
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))

    # 将两张图像水平拼接
    concatenated_image = cv2.hconcat([img1_resized, img2_resized])
    return concatenated_image

