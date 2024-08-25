import numpy as np

def dilate(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    dilated_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if img[i + kernel_center[0], j + kernel_center[1]] == 255:
                dilated_img[i:i_, j:j_] = 255
    return dilated_img[:img_shape[0], :img_shape[1]]

def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 255

    return eroded_img[:img_shape[0], :img_shape[1]]

def open(img, kernel):
    return dilate(erode(img,kernel), kernel)

def close(img, kernel):
    return erode(dilate(img,kernel), kernel)

def bitwise_and(X, Y):
    return np.bitwise_and(np.uint8(X),np.uint8(Y))

def bitwise_or(X, Y):
    return np.bitwise_or(np.uint8(X), np.uint8(Y))

def hitmiss(img, kernel):
    kernel_hit = (kernel == 1).astype(np.uint8)
    kernel_miss = (kernel == -1).astype(np.uint8)

    e1 = erode(img, kernel_hit)
    e2 = erode(255 - img, kernel_miss)
    return bitwise_and(e1, e2)

def extract_boundary(img, kernel):
    return (img-erode(img, kernel))*255.0 #Khi trừ nó lại lấy nhị phân 1, 0 nên kết quả ra 1,0 cần nhân 255

def reconstruct_by_dilation(marker_img, kernel, mask_img):
    prev_img = marker_img
    while True:
        dilated_img = dilate(prev_img, kernel)
        current_img = bitwise_and(dilated_img, mask_img)
        if np.array_equal(prev_img, current_img):
            return prev_img
        prev_img = current_img

def reconstruct_by_erosion(marker_img, kernel, mask_img):
    prev_img = marker_img
    while True:
        eroded_img = erode(prev_img, kernel)
        current_img = bitwise_or(eroded_img, mask_img)
        if np.array_equal(prev_img, current_img):
            return prev_img
        prev_img = current_img

def open_by_reconstruction(marker_img, kernel, mask_img, iterations):
    tmp_img = marker_img
    for _ in range(iterations):
        tmp_img = erode(tmp_img, kernel)
    return reconstruct_by_dilation(tmp_img, kernel, mask_img)

def close_by_reconstruction(marker_img, kernel, mask_img, iterations):
    tmp_img = marker_img
    for _ in range(iterations):
        tmp_img = dilate(tmp_img, kernel)
    return reconstruct_by_erosion(tmp_img, kernel, mask_img)

def fill_hole(img, kernel):
    img_shape = img.shape
    invert_img = np.bitwise_not(img)
    marker_img = np.zeros(img_shape, np.uint8)

    marker_img[:, 0] = 255 - img[:, 0]
    marker_img[:, -1] = 255 - img[:, -1]
    marker_img[0, :] = 255 - img[0, :]
    marker_img[-1, :] = 255 - img[-1, :]

    filled_img = reconstruct_by_dilation(marker_img, kernel, invert_img)
    return np.bitwise_not(filled_img)


def extract_connected_components(img, kernel):
    img_shape = img.shape
    labels = np.zeros(img_shape, np.int32)
    tmp = img.copy()
    component_info = {}
    current_label = 0

    while np.any(tmp != 0):
        current_label += 1
        coords = np.argwhere(tmp == 255)
        x, y = coords[0]
        A = np.zeros(img_shape, np.uint8)
        A[x, y] = 255

        while True:
            B = bitwise_and(dilate(A, kernel), img)
            if np.array_equal(A, B):
                break
            A = B

        component_size = np.sum(A) // 255
        component_info[current_label] = component_size
        tmp = tmp - A
        labels[A != 0] = current_label

    return component_info, labels

def find_convex_hull(points):
    # Convert points to a list of tuples for sorting
    points = [tuple(point) for point in points]
    
    # Sort the points lexicographically (tuples compare like this)
    points = sorted(points)

    # Build the lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build the upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper hull to get the full hull
    return lower[:-1] + upper[:-1]

def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def find_contours(img):
    contours = []
    visited = np.zeros_like(img, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(x, y):
        return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]
    
    def bfs(start):
        queue = [start]
        contour = []
        while queue:
            x, y = queue.pop(0)
            if not visited[x, y]:
                visited[x, y] = True
                contour.append((x, y))
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny) and img[nx, ny] and not visited[nx, ny]:
                        queue.append((nx, ny))
        return contour
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] and not visited[i, j]:
                contour = bfs((i, j))
                contours.append(contour)
    
    return contours

def draw_polylines(image, points, color):
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        rr, cc = draw_line(start, end)
        image[rr, cc] = color

def draw_line(start, end):
    x0, y0 = start
    x1, y1 = end
    rr, cc = [], []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        rr.append(x0)
        cc.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return np.array(rr), np.array(cc)

def convex_hull(img):
    # Find contours
    contours = find_contours(img)
    
    # Find convex hull for each contour
    hulls = [find_convex_hull(contour) for contour in contours]
    
    # Create an empty image to draw the hulls
    # hull_image = np.zeros_like(img)
    
    # Draw the convex hulls
    for hull in hulls:
        hull_points = np.array(hull, dtype=np.int32)
        draw_polylines(img, hull_points, color=255)
    
    return img

def thin(img):
    kernels = [
        np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]]),
        np.array([[0, -1, -1], [1, 1, -1], [1, 1, 0]]),
        np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]]),
        np.array([[1, 1, 0], [1, 1, -1], [0, -1, -1]]),
        np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]]),
        np.array([[0, 1, 1], [-1, 1, 1], [-1, -1, 0]]),
        np.array([[-1, 0, 1], [-1, 1, 1], [-1, 0, 1]]),
        np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 1]])
    ]

    result = img
    for kernel in kernels:
        result = result - hitmiss(result, kernel)

    prev_result = result
    while True:
        for kernel in kernels:
            result = prev_result - hitmiss(prev_result, kernel)
            if np.array_equal(prev_result, result):
                return prev_result
            prev_result = result

def thicken(img):
    kernels = [
        np.array([[1, 1, 1], [0, -1, 0], [-1, -1, -1]]),
        np.array([[0, 1, 1], [-1, -1, 1], [-1, -1, 0]]),
        np.array([[-1, 0, 1], [-1, -1, 1], [-1, 0, 1]]),
        np.array([[-1, -1, 0], [-1, -1, 1], [0, 1, 1]]),
        np.array([[-1, -1, -1], [0, -1, 0], [1, 1, 1]]),
        np.array([[0, -1, -1], [1, -1, -1], [1, 1, 0]]),
        np.array([[1, 0, -1], [1, -1, -1], [1, 0, -1]]),
        np.array([[1, 1, 0], [1, -1, -1], [0, -1, -1]])
    ]

    result = img
    for kernel in kernels:
        result = result + hitmiss(result, kernel)

    return result

def skeleton(img, kernel):
    result = np.zeros_like(img)
    eroded_img = img

    while True:
        opened_img = open(eroded_img, kernel)
        skel_part = eroded_img - opened_img
        result = result + skel_part
        eroded_img = erode(eroded_img, kernel)
        if np.all(eroded_img == 0):
            break

    return result

def subtract(img1, img2):
    return np.clip(img1 - img2, 0, None)

def skeletonize(img, kernel):
    skel = np.zeros(img.shape, np.uint8)
    element = kernel

    while True:
        eroded = erode(img, element)
        temp = dilate(eroded, element)
        temp = subtract(img, temp)
        skel = bitwise_or(skel, temp)
        img = eroded.copy()
        if np.count_nonzero(img) == 0:
            break

    return skel

def prun(img):
    kernels = [
        np.array([[0, -1, -1], [1, 1, -1], [0, -1, -1]]),
        np.array([[0, 1, 0], [-1, 1, -1], [-1, -1, -1]]),
        np.array([[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]]),
        np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]]),
        np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
        np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]),
        np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
        np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]])
    ]

    result = img
    for kernel in kernels:
        result = result - hitmiss(result, kernel)

    pruned_result = np.zeros_like(img)
    for kernel in kernels:
        pruned_result += hitmiss(result, kernel)

    return result + bitwise_and(dilate(pruned_result, np.ones((3, 3))), img)
