import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt

import PIL
from PIL import Image
from PIL import ImageDraw
import scipy
import scipy.misc
from scipy import ndimage
import time
import cv2
cv2.useOptimized()


def save_metadata(metadata, contour_path, batch_id):
    # Converts metadata (list of lists) into an nparray, and then saves
    metadata_path = os.path.join(contour_path, 'metadata')
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    metadata_fn = str(batch_id) + '.npy'
    np.save(os.path.join(metadata_path,metadata_fn), metadata)


# Accumulate metadata
def accumulate_meta(array, subpath, filename, args, nimg, paddle_margin = None):
    if paddle_margin is None:
        # OLD VERSION
        array += [[subpath, filename, nimg,
                   args.continuity, args.contour_length, args.distractor_length, args.num_distractor_contours,
                   args.paddle_length, args.paddle_thickness, args.paddle_margin]]
    else:
        # NEW VERSION
        array += [[subpath, filename, nimg,
                   args.continuity, args.contour_length, args.distractor_length, args.num_distractor_contours,
                   args.paddle_length, args.paddle_thickness, paddle_margin, len(args.paddle_contrast_list)]]
    return array
    # GENERATED ARRAY IS NATURALLY SORTED BY THE ORDER IN WHICH IMGS ARE CREATED.
    # IN TRAIN OR TEST TIME CALL np.random.shuffle(ARRAY)

    
def make_many_snakes(image, mask,
                     num_snakes, max_snake_trial,
                     num_segments, segment_length, thickness, margin, continuity,
                     contrast_list,
                     max_segment_trial,
                     aa_scale,
                     display_final = False, display_snake = False, display_segment = False,
                     allow_incomplete=False,
                     allow_shorter_snakes=False,
                     stop_with_availability=None, 
                     fn_draw_circle=None):
    curr_image = image.copy()
    curr_mask = mask.copy()
    isnake = 0

    small_dilation_structs = generate_dilation_struct(margin)
    large_dilation_structs = generate_dilation_struct(margin*aa_scale)
    
    assert type(num_segments) == int, f'{num_segments}'
    
    if image is None:
        print('No image. Previous run probably failed.')
    while isnake < num_snakes:
        snake_retry_count = 0
        while snake_retry_count <= max_snake_trial:
            curr_image, curr_mask, success = \
            make_snake(curr_image, curr_mask,
                       num_segments, segment_length, thickness, margin, continuity, small_dilation_structs, large_dilation_structs,
                       contrast_list,
                       max_segment_trial,
                       aa_scale, display_snake, display_segment, allow_shorter_snakes, stop_with_availability,
                       fn_draw_circle=fn_draw_circle)
            if success is False:
                snake_retry_count += 1
            else:
                break
        if snake_retry_count > max_snake_trial:
            #print('Exceeded max # of snake re-rendering.')
            if not allow_incomplete:
                print('Required # snakes unmet. Aborting')
                return None, None
        isnake += 1
    if display_final:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(curr_image)
        plt.subplot(1, 2, 2)
        plt.imshow(curr_mask)
        plt.show()
    return curr_image, curr_mask


def find_available_coordinates(mask, margin):
    if np.min(mask)<0:
        #print('mystery')
        return (np.array([]),np.array([])), 0
    # get temporarily dilated mask
    if margin>0:
        dilated_mask = mask.copy() # TODO: TURNED OFF ANYWAY
        #dilated_mask = binary_dilate(mask, margin, type='1', scale=1.)
    elif margin == 0:
        dilated_mask = mask.copy()
    # get a list of available coordinates
    available_coordinates = np.nonzero(1-dilated_mask.astype(np.uint8))
    num_available_coordinates = available_coordinates[0].shape[0]
    return available_coordinates, num_available_coordinates


def make_snake(image, mask,
               num_segments, segment_length, thickness, margin, continuity, small_dilation_structs, large_dilation_structs,
               contrast_list,
               max_segment_trial, aa_scale,
               display_snake = False, display_segment = False,
               allow_shorter_snakes=False, stop_with_availability=None, 
               fn_draw_circle=None):
    # set recurring state variables
    current_segment_mask = np.zeros_like(mask)
    current_image = image.copy()
    current_mask = mask.copy()
    # draw initial segment
    for isegment in range(1):
        num_possible_contrasts = len(contrast_list)
        if num_possible_contrasts>0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = contrast_list[contrast_index]
        current_image, current_mask, current_segment_mask, current_pivot, current_orientation, success \
        = seed_snake(current_image, current_mask,
                     max_segment_trial,segment_length, thickness, margin, contrast,  small_dilation_structs, large_dilation_structs,
                     aa_scale= aa_scale, display=display_segment, stop_with_availability=stop_with_availability)
        if fn_draw_circle is not None and current_pivot is not None:
            current_image = np.maximum(current_image, fn_draw_circle(current_pivot).astype(current_image.dtype))
        if success is False:
            return image, mask, False
    # sequentially add segments
    for isegment in range(num_segments-1):
        if num_possible_contrasts>0:
            contrast_index = np.random.randint(low=0, high=num_possible_contrasts)
        else:
            contrast_index = 0
        contrast = contrast_list[contrast_index]
        current_image, current_mask, current_segment_mask, current_pivot, current_orientation, _, success \
        = extend_snake(list(current_pivot), current_orientation, current_segment_mask,
                         current_image, current_mask, max_segment_trial,
                         segment_length, thickness, margin, continuity, contrast,  small_dilation_structs, large_dilation_structs,
                         aa_scale = aa_scale,
                         display=display_segment,
                         forced_current_pivot=None)
        if success is False:
            if allow_shorter_snakes:
                return image, mask, True
            else:
                return image, mask, False
    current_mask = np.maximum(current_mask, current_segment_mask)
    # display snake
    if display_snake:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(current_image)
        plt.subplot(1, 2, 2)
        plt.imshow(current_mask)
        plt.show()
    return current_image, current_mask, True


# last_pivot: coordinate of the anchor of two segments ago
def seed_snake(image, mask,
               max_segment_trial, length, thickness, margin, contrast,  small_dilation_structs, large_dilation_structs,
               aa_scale, display = False, stop_with_availability=None):

    struct_shape = ((length+margin)*2+1, (length+margin)*2+1)
    struct_head = [length+margin+1, length+margin+1]

    trial_count = 0
    while trial_count <= max_segment_trial:
        sampled_orientation_in_rad = np.random.randint(low=-180, high=180) * np.pi / 180
        if sampled_orientation_in_rad+np.pi < np.pi:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad + np.pi
        else:
            sampled_orientation_in_rad_reversed = sampled_orientation_in_rad - np.pi

        # generate dilation struct
        _, struct = draw_line_n_mask(struct_shape, struct_head, sampled_orientation_in_rad, length, thickness, margin, large_dilation_structs, aa_scale)
            # head-centric struct

        # dilate mask using segment
        lined_mask = mask.copy()
        lined_mask[0,:] = 1
        lined_mask[-1,:] = 1
        lined_mask[:,0] = 1
        lined_mask[:,-1] = 1
        dilated_mask = binary_dilate_custom(lined_mask, struct, value_scale=1.)
            # dilation in the same orientation as the tail

        # run coordinate searcher while also further dilating
        _, raw_num_available_coordinates = find_available_coordinates(np.ceil(mask-0.3), margin=0)
        available_coordinates, num_available_coordinates = find_available_coordinates(np.ceil(dilated_mask-0.3), margin=0)
        if (stop_with_availability is not None) & \
           (np.float(raw_num_available_coordinates)/(mask.shape[0]*mask.shape[1]) < stop_with_availability):
            #print('critical % of mask occupied before dilation. finalizing')
            return image, mask, np.zeros_like(mask), None, None, False
        if num_available_coordinates == 0:
            #print('Mask fully occupied after dilation. finalizing')
            return image, mask, np.zeros_like(mask), None, None, False
            continue

        # sample coordinate and draw
        random_number = np.random.randint(low=0,high=num_available_coordinates)
        sampled_tail = [available_coordinates[0][random_number],available_coordinates[1][random_number]] # CHECK OUT OF BOUNDARY CASES
        sampled_head = translate_coord(sampled_tail, sampled_orientation_in_rad, length)
        sampled_pivot = translate_coord(sampled_head, sampled_orientation_in_rad_reversed, length+margin)
        if (sampled_head[0] < 0) | (sampled_head[0] >= mask.shape[0]) | \
           (sampled_head[1] < 0) | (sampled_head[1] >= mask.shape[1]) | \
           (sampled_pivot[0] < 0) | (sampled_pivot[0] >= mask.shape[0]) | \
           (sampled_pivot[1] < 0) | (sampled_pivot[1] >= mask.shape[1]):
            #print('missampled seed +segment_trial_count')
            trial_count += 1
            continue
        else:
            break
    if trial_count > max_segment_trial:
        return image, mask, np.zeros_like(mask), None, None, False

    l_im, m_im = draw_line_n_mask((mask.shape[0], mask.shape[1]), sampled_tail, sampled_orientation_in_rad, length, thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)
    image = np.maximum(image, l_im)

    if display:
        plt.figure(figsize=(10,20))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(dilated_mask)
        plt.title(str(num_available_coordinates))
        plt.plot(sampled_tail[1], sampled_tail[0], 'bo')
        plt.plot(sampled_head[1], sampled_head[0], 'ro')
        plt.show()
    return image, mask, m_im, sampled_pivot, sampled_orientation_in_rad, True


# last_pivot: coordinate of the anchor of two segments ago
def extend_snake(last_pivot, last_orientation, last_segment_mask,
                 image, mask, max_segment_trial,
                 length, thickness, margin, continuity, contrast, small_dilation_structs, large_dilation_structs,
                 aa_scale,
                 display = False,
                 forced_current_pivot=None):
    # set anchor
    if forced_current_pivot is not None:
        new_pivot = list(forced_current_pivot)
    else:
        new_pivot = translate_coord(last_pivot, last_orientation, length+2*margin)
    # get temporarily dilated mask
    dilated_mask = binary_dilate_custom(mask, small_dilation_structs, value_scale=1.)
    # get candidate endpoints
    unique_coords, unique_orientations, cmf, pmf = get_coords_cmf(new_pivot, last_orientation, length+margin, dilated_mask, continuity)
    # sample endpoint
    if pmf is None:
        return image, mask, None, None, None, None, False
    else:
        trial_count = 0
        segment_found = False
        while trial_count <= max_segment_trial:
            random_num = np.random.rand()
            sampled_index = np.argmax(cmf - random_num > 0)
            new_orientation = unique_orientations[sampled_index]
            new_head = unique_coords[sampled_index, :]  # find the smallest index whose value is greater than rand
            flipped_orientation = flip_by_pi(new_orientation)
            l_im, m_im = draw_line_n_mask((mask.shape[0],mask.shape[1]), new_head, flipped_orientation, length, thickness, margin, large_dilation_structs, aa_scale, contrast_scale=contrast)

            trial_count += 1
            if np.max(mask+m_im) < 1.8:
                segment_found = True
                break
            else:
                continue
            ## TODO: ADD JITTERED ORIENTATION
        if segment_found == False:
            # print('extend_snake: self-crossing detected')
            # print('pmf of sample ='+str(pmf[0,sampled_index]))
            # print('mask at sample ='+str(dilated_mask[new_head[0],new_head[1]]))
            # print('smaple ='+str(new_head))
            image = np.maximum(image, l_im)
            if display:
                plt.subplot(1, 4, 1)
                plt.imshow(image)
                plt.subplot(1, 4, 2)
                plt.imshow(mask)
                plt.subplot(1, 4, 3)
                plt.imshow(dilated_mask)
                plt.subplot(1, 4, 4)
                plt.imshow(mask + m_im)
                plt.plot(new_pivot[1],new_pivot[0], 'go')
                plt.plot(new_head[1], new_head[0], 'ro')
                plt.show()
            return image, mask, None, None, None, None, False
        else:
            image = np.maximum(image, l_im)
            mask = np.maximum(mask, last_segment_mask)
            if display:
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.subplot(1, 2, 2)
                plt.imshow(mask)
                plt.plot(new_pivot[1],new_pivot[0], 'go')
                plt.plot(new_head[1], new_head[0], 'ro')
                plt.show()
            return image, mask, m_im, new_pivot, new_orientation, new_head, True


# given the coordinate (y,x) of the last_endpoint and a fixed radius and a mask,
# returns a list of coordinates that are nearest neighbors around a circle cetnered at (y,x) and of radius rad.
# Also returns the CMF at each of the returned coordinates.
# continuity : parameter controlling continuity. (minimum 1.0 -> flatline at 90 degs)
def get_coords_cmf(last_endpoint, last_orientation, step_length, mask, continuity):
    height = mask.shape[0]
    width = mask.shape[1]
    # compute angle of an arc whose length equals to the size of one pixel
    deg_per_pixel = 360. / (2 * step_length * np.pi)
    samples_in_rad = np.arange(0, 360, deg_per_pixel) * np.pi / 180
    # get lists of y and x coordinates (exclude out-of-bound coordinates)
    samples_in_y = last_endpoint[0] + (step_length * np.sin(samples_in_rad))
    samples_in_x = last_endpoint[1] + (step_length * np.cos(samples_in_rad))
    samples_in_coord = np.concatenate((np.expand_dims(samples_in_y, axis=1),
                                       np.expand_dims(samples_in_x, axis=1)), axis=1).astype(int)
    OOB_rows = (samples_in_y>=height) | (samples_in_y<0) | (samples_in_x>=width) | (samples_in_x<0)
    samples_in_coord = np.delete(samples_in_coord, np.where(OOB_rows), axis=0)
    if samples_in_coord.shape[0]==0:
        #print('dead-end while expanding')
        return None, None, None, None
    # find unique coordinates and related quantities
    unique_coords, indices = np.unique(samples_in_coord, axis=0, return_index=True)
    unique_displacements = unique_coords - last_endpoint
    unique_orientations = np.arctan2(unique_displacements[:,0], unique_displacements[:,1]) # from -pi to pi
    unique_delta = np.minimum(np.abs(unique_orientations - last_orientation),
                              2*np.pi - np.abs(unique_orientations - last_orientation))
    unique_delta = np.minimum(unique_delta*continuity,
                              0.5*np.pi)
    unique_cosinedistweights = np.maximum(np.cos(unique_delta), 0)**2
    # compute probability distribution
    inverted = 1 - mask[unique_coords[:, 0], unique_coords[:, 1]]
    pmf = np.multiply(np.array([inverted]),unique_cosinedistweights) # weight coordinates according to continuity
    total = np.sum(pmf)
    if total < 1e-4:
        #print('dead-end while expanding')
        return None, None, None, None
    pmf = pmf / total
    cmf = np.cumsum(pmf)
    return unique_coords, unique_orientations, cmf, pmf


def draw_line_n_mask(im_size, start_coord, orientation, length, thickness, margin, large_dilation_struct, aa_scale, contrast_scale=1.0):
    # sanity check
    if np.round(thickness*aa_scale) - thickness*aa_scale != 0.0:
        raise ValueError('thickness does not break even.')

    # draw a line in a finer resolution
    miniline_blown_shape = (length + int(np.ceil(thickness)) + margin) * 2 * aa_scale + 1
    miniline_blown_center = (length + int(np.ceil(thickness)) + margin) * aa_scale
    miniline_blown_thickness = int(np.round(thickness*aa_scale))
    miniline_blown_head = translate_coord([miniline_blown_center, miniline_blown_center], orientation, length*aa_scale)
    miniline_blown_im = Image.new('F', (miniline_blown_shape, miniline_blown_shape), 'black')
    line_draw = ImageDraw.Draw(miniline_blown_im)
    line_draw.line([(miniline_blown_center, miniline_blown_center),
                    (miniline_blown_head[1],miniline_blown_head[0])],
                   fill='white', width=miniline_blown_thickness)

    # resize with interpolation + apply contrast
    miniline_shape = (length + int(np.ceil(thickness)) + margin) *2 + 1
    # miniline_im = scipy.misc.imresize(np.array(miniline_blown_im),
    #                                   (miniline_shape, miniline_shape),
    #                                   interp='lanczos').astype(np.float)/255
    
    miniline_im = np.array(miniline_blown_im.resize(
        (miniline_shape, miniline_shape), PIL.Image.LANCZOS)).astype(np.float)/255
    
    if contrast_scale != 1.0:
        miniline_im *= contrast_scale

    # draw a mask
    minimask_blown_im = binary_dilate_custom(miniline_blown_im, large_dilation_struct, value_scale=1.).astype(np.uint8)
    # minimask_im = scipy.misc.imresize(np.array(minimask_blown_im),
                        # (miniline_shape, miniline_shape),
                        # interp='lanczos').astype(np.float) / 255

    minimask_im = np.array(Image.fromarray(minimask_blown_im).resize(
        (miniline_shape, miniline_shape), PIL.Image.LANCZOS)).astype(np.float)/255
    
    # place in original shape
    l_im = np.array(Image.new('F', (im_size[1], im_size[0]), 'black'))
    m_im = l_im.copy()
    l_im_vertical_range_raw = [start_coord[0] - (length + int(np.ceil(thickness)) + margin),
                               start_coord[0] + (length + int(np.ceil(thickness)) + margin)]
    l_im_horizontal_range_raw = [start_coord[1] - (length + int(np.ceil(thickness)) + margin),
                                 start_coord[1] + (length + int(np.ceil(thickness)) + margin)]
    l_im_vertical_range_rectified = [np.maximum(l_im_vertical_range_raw[0], 0),
                                     np.minimum(l_im_vertical_range_raw[1], im_size[0]-1)]
    l_im_horizontal_range_rectified = [np.maximum(l_im_horizontal_range_raw[0], 0),
                                       np.minimum(l_im_horizontal_range_raw[1], im_size[1]-1)]
    miniline_im_vertical_range_rectified = [np.maximum(0,-l_im_vertical_range_raw[0]),
                                            miniline_shape - 1 - np.maximum(0,l_im_vertical_range_raw[1]-(im_size[0]-1))]
    miniline_im_horizontal_range_rectified = [np.maximum(0,-l_im_horizontal_range_raw[0]),
                                              miniline_shape - 1 - np.maximum(0,l_im_horizontal_range_raw[1]-(im_size[1]-1))]
    l_im[l_im_vertical_range_rectified[0]:l_im_vertical_range_rectified[1]+1,
         l_im_horizontal_range_rectified[0]:l_im_horizontal_range_rectified[1]+1] = \
        miniline_im[miniline_im_vertical_range_rectified[0]:miniline_im_vertical_range_rectified[1] + 1,
                    miniline_im_horizontal_range_rectified[0]:miniline_im_horizontal_range_rectified[1] + 1].copy()
    m_im[l_im_vertical_range_rectified[0]:l_im_vertical_range_rectified[1]+1,
         l_im_horizontal_range_rectified[0]:l_im_horizontal_range_rectified[1]+1] = \
        minimask_im[miniline_im_vertical_range_rectified[0]:miniline_im_vertical_range_rectified[1] + 1,
                    miniline_im_horizontal_range_rectified[0]:miniline_im_horizontal_range_rectified[1] + 1].copy()

    return l_im, m_im


def binary_dilate_custom(im, struct, value_scale=1.):
    out = np.array(cv2.dilate(np.array(im), kernel=struct.astype(np.uint8), iterations = 1)).astype(float)/value_scale
    return out


def generate_dilation_struct(margin):
    kernel = np.zeros((2 * margin + 1, 2 * margin + 1))
    y, x = np.ogrid[-margin:margin + 1, -margin:margin + 1]
    mask = x ** 2 + y ** 2 <= margin ** 2
    kernel[mask] = 1
    return kernel


# translate a coordinate. orientation is in radian.
# if allow_float, function returns exact coordinate (not rounded)
def translate_coord(coord, orientation, dist, allow_float=False):
    y_displacement = float(dist)*np.sin(orientation)
    x_displacement = float(dist)*np.cos(orientation)
    if allow_float is True:
        new_coord = [coord[0]+y_displacement, coord[1]+x_displacement]
    else:
        new_coord = [int(np.ceil(coord[0] + y_displacement)), int(np.ceil(coord[1] + x_displacement))]
    return new_coord


# flip an orientation by pi. All in RADIAN
def flip_by_pi(orientation):
    if orientation<0:
        flipped_orientation = orientation + np.pi
    else:
        flipped_orientation = orientation - np.pi
    return flipped_orientation


# Turn a grayscale (np array) image into an RGB in red channel
def gray2red(im, bw='w'):
    if bw == 'b':
        padding = np.zeros((im.shape[0],im.shape[1],2))
        im_expanded = np.expand_dims(im,axis=-1)
        out = np.concatenate([im_expanded,padding],axis=-1)/255
    elif bw == 'w':
        padding = 255*np.ones((im.shape[0], im.shape[1], 3))
        im_expanded = np.tile(im, reps=(3,1,1))
        im_expanded = np.transpose(im_expanded, (1,2,0))
        im_expanded[:,:,0] = 0
        out = (padding - im_expanded)/255
    return out


# Turn a grayscale image into an RGB in gray
def gray2gray(im, bw='w'):
    if bw == 'b':
        im_expanded = np.tile(im, reps=(3,1,1))
        im_expanded = np.transpose(im_expanded, (1,2,0))
    elif bw == 'w':
        im_expanded = np.tile(im, reps=(3,1,1))
        im_expanded = np.transpose(im_expanded, (1,2,0))
        im_expanded = 255-im_expanded
    return im_expanded / 255

# Sum image
def imsum(im1, im2, bw='w'):
    if bw == 'b':
        out = np.maximum(im1, im2)
    elif bw == 'w':
        out = np.minimum(im1 , im2)
    return out


# ALGORITHM
# 1. compute initial point
#    current_start = translate(last_endpoint, last_orientation, dilation+1)
# 2. draw current_endpoint (distance = line_length + dilation)
#    compute current_orientation
#    M' <--- dilate(M, dilation+2)
#    sample endpoint using M'
#    trial_count += 1
# 3. compute line and mask
#    l_current, m_current = draw_line_n_mask(translate(current_start, current_orientation, dilation), current_endpoint, dilation)
# 4. check if max(M + m_current) > 2
#       yes -> check if retrial_count > max_count
#           yes -> return with failure flag
#           no -> goto 2
#       no -> goto 5
# 5. draw image I += l_current
# 6. draw mask M = max(M, m_last)
# 7. m_last = m_current.copy()
# 8. retrial_count = 0



# def test():
#     t = time.time()

#     target_paddle_length =12  # from 6 to 18
#     distractor_paddle_length = target_paddle_length / 3
#     num_distractor_paddles = int(33*(9./target_paddle_length)) #4
#     continuity = 2.4 #1  # from 1 to 2.5 (expect occasional failures at high values)

#     imsize = 256
#     aa_scale = 4
#     segment_length = 5
#     thickness = 1.5
#     contrast_list = [1.0]
#     margin = 4

#     image = np.zeros((imsize, imsize))
#     mask = np.zeros((imsize, imsize))

#     ### TODO: missampled seed + % occupied constraint

#     num_segments = target_paddle_length
#     num_snakes = 1
#     max_snake_trial = 10
#     max_segment_trial = 2
#     image1, mask = make_many_snakes(image, mask,
#                                     num_snakes, max_snake_trial,
#                                     num_segments, segment_length, thickness, margin, continuity, contrast_list,
#                                     max_segment_trial, aa_scale,
#                                     display_final=False, display_snake=False, display_segment=False,
#                                     allow_incomplete=False, allow_shorter_snakes=False)
#     num_segments = distractor_paddle_length
#     num_snakes = num_distractor_paddles
#     max_snake_trial = 4
#     max_segment_trial = 2
#     image2, mask = make_many_snakes(image1, mask,
#                                     num_snakes, max_snake_trial,
#                                     num_segments, segment_length, thickness, margin, continuity, contrast_list,
#                                     max_segment_trial, aa_scale,
#                                     display_final=False, display_snake=False, display_segment=False,
#                                     allow_incomplete=False, allow_shorter_snakes=False)
#     num_segments = 1
#     num_snakes = 0 #400 - target_paddle_length - num_distractor_paddles * distractor_paddle_length
#     max_snake_trial = 3
#     max_segment_trial = 2
#     image3, _ = make_many_snakes(image2, mask,
#                                  num_snakes, max_snake_trial,
#                                  num_segments, segment_length, thickness, margin, continuity, contrast_list,
#                                  max_segment_trial, aa_scale,
#                                  display_final=False, display_snake=False, display_segment=False,
#                                  allow_incomplete=True, allow_shorter_snakes=False, stop_with_availability=0.01)

#     plt.figure(figsize=(10, 10))
#     plt.subplot(2, 1, 1)
#     red_target = gray2red(image1)
#     show1 = scipy.misc.imresize(red_target, (imsize, imsize), interp='lanczos')
#     plt.imshow(show1)
#     plt.axis('off')

#     plt.subplot(2, 1, 2)
#     gray_total = gray2gray(1 - image3)
#     show2 = scipy.misc.imresize(gray_total, (imsize, imsize), interp='lanczos')
#     plt.imshow(show2)
#     plt.axis('off')

#     elapsed = time.time() - t
#     print('ELAPSED TIME : ', str(elapsed))

#     plt.show()

# def from_wrapper(args):

#     t = time.time()
#     iimg = 0

#     if (args.save_images):
#         contour_sub_path = os.path.join('imgs', str(args.batch_id))
#         if not os.path.exists(os.path.join(args.contour_path, contour_sub_path)):
#             os.makedirs(os.path.join(args.contour_path, contour_sub_path))
#     if (args.save_gt):
#         gt_sub_path = os.path.join('gt_imgs', str(args.batch_id))
#         if not os.path.exists(os.path.join(args.contour_path, gt_sub_path)):
#             os.makedirs(os.path.join(args.contour_path, gt_sub_path))
#     if args.save_metadata:
#         metadata = []
#         # CHECK IF METADATA FILE ALREADY EXISTS
#         metadata_path = os.path.join(args.contour_path, 'metadata')
#         if not os.path.exists(metadata_path):
#             os.makedirs(metadata_path)
#         metadata_fn = str(args.batch_id) + '.npy'
#         metadata_full = os.path.join(metadata_path, metadata_fn)
#         if os.path.exists(metadata_full):
#             print('Metadata file already exists.')
#             return

#     while (iimg < args.n_images):
#         print('Image# : %s'%(iimg))

#         # Sample paddle margin
#         num_possible_margins = len(args.paddle_margin_list)
#         if num_possible_margins > 0:
#             margin_index = np.random.randint(low=0, high=num_possible_margins)
#         else:
#             margin_index = 0
#         margin = args.paddle_margin_list[margin_index]
#         base_num_paddles = 400
#         num_paddles_factor = 1./((7.5 + 13*margin + 4*margin*margin)/123.5)
#         total_num_paddles = int(base_num_paddles*num_paddles_factor)

#         image = np.zeros((args.window_size[0], args.window_size[1]))
#         mask = np.zeros((args.window_size[0], args.window_size[1]))
#         target_im, mask = make_many_snakes(image, mask,
#                                            1, args.max_target_contour_retrial,
#                                            args.contour_length, args.paddle_length, args.paddle_thickness, margin, args.continuity, args.paddle_contrast_list,
#                                            args.max_paddle_retrial, args.antialias_scale,
#                                            display_final=False, display_snake=False, display_segment=False,
#                                            allow_incomplete=False, allow_shorter_snakes=False)
#         if (target_im is None):
#             continue
#         interm_im, mask = make_many_snakes(target_im, mask,
#                                            args.num_distractor_contours, args.max_distractor_contour_retrial,
#                                            args.distractor_length, args.paddle_length, args.paddle_thickness, margin, args.continuity, args.paddle_contrast_list,
#                                            args.max_paddle_retrial, args.antialias_scale,
#                                            display_final=False, display_snake=False, display_segment=False,
#                                            allow_incomplete=False, allow_shorter_snakes=False)
#         if (interm_im is None):
#             continue
#         if args.use_single_paddles is not False:
#             num_bits = args.use_single_paddles - args.contour_length - args.distractor_length * args.num_distractor_contours
#             final_im, mask = make_many_snakes(interm_im, mask,
#                                               num_bits, 10,
#                                               1, args.paddle_length, args.paddle_thickness, margin, args.continuity, args.paddle_contrast_list,
#                                               args.max_paddle_retrial, args.antialias_scale,
#                                               display_final=False, display_snake=False, display_segment=False,
#                                               allow_incomplete=True, allow_shorter_snakes=False, stop_with_availability=0.01)
#             if (final_im is None):
#                 continue
#         else:
#             final_im = interm_im

#         if (args.pause_display):
#             plt.figure(figsize=(10, 10))
#             plt.subplot(2, 1, 1)
#             red_target = gray2red(1 - target_im)
#             plt.imshow(red_target)
#             plt.axis('off')
#             plt.subplot(2, 1, 2)
#             gray_total = gray2gray(final_im)
#             plt.imshow(gray_total)
#             plt.axis('off')
#             plt.show()

#         if (args.save_images):
#             fn = "sample_%s.png"%(iimg)
#             scipy.misc.imsave(os.path.join(args.contour_path, contour_sub_path, fn), final_im)
#         if (args.save_gt):
#             fn = "gt_%s.png"%(iimg)
#             scipy.misc.imsave(os.path.join(args.contour_path, gt_sub_path, fn), target_im)
#         if (args.save_metadata):
#             metadata = accumulate_meta(metadata, contour_sub_path, fn, args, iimg, paddle_margin=margin)
#             ## TODO: GT IS NOT INCLUDED IN METADATA
#         iimg += 1

#     if (args.save_metadata):
#         matadata_nparray = np.array(metadata)
#         save_metadata(matadata_nparray, args.contour_path, args.batch_id)
#     elapsed = time.time() - t
#     print('ELAPSED TIME : ', str(elapsed))

#     plt.show()

#     return

# if __name__ == "__main__":
#     test()


