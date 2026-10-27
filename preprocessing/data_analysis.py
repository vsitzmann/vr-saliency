from __future__ import print_function

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import scipy
from scipy.ndimage import filters
from scipy import misc
from scipy import stats

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from matplotlib.pyplot import plot, draw, show
from matplotlib import animation

import glob
import matplotlib.cm as cm

import msgpack as msg
import msgpack_numpy
msgpack_numpy.patch()
import cPickle as pickle
import preprocessing

import fixation_detector

from settings import *

fov_vert = 106.188
aspect_ratio = 0.82034051
fov_hor = fov_vert * aspect_ratio

np.set_printoptions(threshold=np.inf)
float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

matplotlib.rcParams['figure.dpi'] = 80   # default = 80
matplotlib.rcParams['savefig.dpi'] = 80  # default = 100

plt.style.use('ggplot')
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

def lat_long_to_yx(lat_long, img_height, img_width):
    return (lat_long / np.array([180.0, 360.0])) * (img_height, img_width)


def combined_heatmap(list_of_logs):
    max_length = max([len(experiment_data['timestamp'][::10]) for experiment_data in list_of_logs])

    img_height, img_width = 256, 512

    eye_fixation_coord_array = np.zeros((len(list_of_logs), max_length, 3))
    for idx, experiment_data in enumerate(list_of_logs):
        temp_array = np.column_stack([lat_long_to_yx(experiment_data['gaze_lat_lon'][::10], img_height, img_width),
                                     experiment_data['gaze_fixations'][::10, 0]])
        eye_fixation_coord_array[idx, :len(temp_array)] = temp_array

    fig,ax = plt.subplots(1)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', size='x-large', transform=ax.transAxes)

    combined_heatmap.heatmap = np.zeros((img_height, img_width)).astype(float)
    combined_heatmap.gaze_counts = np.zeros((img_height, img_width))

    im = plt.imshow(combined_heatmap.heatmap, animated=True, vmin=0.0, vmax=1.0, interpolation=None)

    def init():
        time_text.set_text('')
        combined_heatmap.heatmap = np.zeros((img_height, img_width)).astype(float)
        combined_heatmap.gaze_counts = np.zeros((img_height, img_width))
        im.set_array(np.zeros((img_height, img_width)))
        return [im] + [time_text]

    def update_heatmap(i):
        for eye_coords in eye_fixation_coord_array[:, i, :]:
            # If this run is over or this coordinate is not a fixation, end
            if np.allclose(eye_coords, np.array([0.0, 0.0, 0.0])) or eye_coords[2]==0: continue
            combined_heatmap.gaze_counts[eye_coords[0], eye_coords[1]] += 1.0

        time_text.set_text(time_template % (i * 1.0/12.0))

        sigma = img_width/360.0
        combined_heatmap.heatmap = filters.gaussian_filter(combined_heatmap.gaze_counts, sigma=sigma, mode='wrap')
        combined_heatmap.heatmap /= np.maximum(0.01, np.amax(combined_heatmap.heatmap))

        im.set_array(combined_heatmap.heatmap)
        return [im] + [time_text]

    ani = animation.FuncAnimation(fig, update_heatmap, init_func=init, frames=max_length,
                                  interval=100.0/120.0, blit=True)
    plt.show(block=False)


def filter_starting_point(list_of_runs, threshold_deg=20.0):
    cleaned_runs = []
    for run in list_of_runs:
        init_starting_point_lon = run['gaze_lat_lon'][0, 1]
        outside_bool = np.absolute(init_starting_point_lon - run['gaze_lat_lon'][:, 1])>threshold_deg
        if np.any(outside_bool):
            first_left = np.amin(np.where(outside_bool)[0])
            cleaned_runs.append({key:value[first_left:] for key, value in run.iteritems()})
        else:
            cleaned_runs.append(run)
    return cleaned_runs


def filter_starting_point_viewpoint_known(list_of_logs, viewpoint_idx, threshold_deg=20.0):
    cleaned_runs = []
    init_starting_point_lon = [0.0, 90.0, 180.0, 270.0][viewpoint_idx]
    for run in list_of_logs:
        first_left = np.min(np.where(np.absolute(init_starting_point_lon - run['gaze_lat_lon'][:, 1])>threshold_deg)[0])
        cleaned_runs.append({key:value[first_left:] for key, value in run.iteritems()})
    return cleaned_runs


def get_stats(list_of_runs):
    print("\n\n********Statistics for %d runs********"%len(list_of_runs))
    fixation_idcs = np.vstack([run['gaze_fixations'] for run in list_of_runs])[:, 1] != 0
    for column in ['headLatLon', 'gaze_lat_lon_offset', 'gaze_lat_lon', 'ang_head_velo']:
        all_values = np.vstack([run[column] for run in list_of_runs])

        if 'offset' in column or 'velo' in column:
            all_values = np.absolute(all_values)

        print("*******%s*******"%column)
        print("Mean: %s"%np.mean(all_values, axis=0))
        print("Median: %s"%np.median(all_values, axis=0))
        print("Std: %s"%np.std(all_values, axis=0))
        print("Max: %s"%np.amax(all_values, axis=0))
        print("Min: %s"%np.amin(all_values, axis=0))
        print("Mean during fixations: %s"%np.mean(all_values[fixation_idcs], axis=0))
        print("Std during fixations: %s"%np.std(all_values[fixation_idcs], axis=0))
        print('\n')

    all_fixations = np.vstack([run['gaze_fixations'] for run in list_of_runs])
    fixation_uniques = np.unique(all_fixations[all_fixations[:, 1]!=0, 1])

    all_times = []
    for unique in fixation_uniques:
        begin_index = np.amin(np.where(all_fixations[:, 1]==unique)[0])
        end_index = np.amax(np.where(all_fixations[:, 1]==unique)[0])

        all_times.append(end_index - begin_index)

    print('\n')
    print("*******Fixations*******")
    print("Mean duration: %s"%np.mean(all_times))
    print("Std duration: %s"%np.std(all_times))
    print("Max duration: %s"%np.amax(all_times, axis=0))
    print('\n')


def print_logfile_stats(log):
    print("Total of %d runs."%len(log))

    for i in xrange(4):
        print("Viewpoint %d: %d"%(i, len(log[log['viewpoint_idx']==i])))


def load_logfile(path):
    with open(path, 'r') as log_file:
        log = pickle.load(log_file)
    print("Dataset loaded.")
    print_logfile_stats(log)
    return log


def get_eye_fixations(log):
    temp_df = pd.DataFrame()
    temp_df['index'] = np.arange(len(log['timestamp']))
    temp_df['timestamp'] = log['timestamp']

    temp_df['norm_pos'] = lat_long_to_yx(log['gaze_lat_lon'], 1.0, 1.0)[:, [1, 0]].tolist()
    temp_df['confidence'] = log['both_confidence']

    temp_df['fixation'] = np.zeros(len(log['timestamp']))
    curr_run_dict = temp_df.to_dict('records')

    fix_dect = fixation_detector.Fixation_Detector_Dispersion_Duration(h_fov=360.0, v_fov=180.0)
    fixations_dict = fix_dect.classify(curr_run_dict)

    fixations_result = np.zeros((len(log['timestamp']), 2))
    for fixation in fixations_dict:
        fixations_result[fixation['start_frame_index']:fixation['end_frame_index']] = fixation['norm_pos']

    return fixations_result, fixations_dict


def filter_viewpoint(log, viewpoint_idx):
    return log[log['viewpoint_idx']==viewpoint_idx]


def visualize_log(log):
    norm_fixation_coords = get_eye_fixations(log)
    overlay_static_coords(norm_fixation_coords, '/media/vsitzmann/Transcend/SIGGRAPH_dataset/Experiments/cubemap_0000/panorama.png')
    heatmap_from_norm_yx(norm_fixation_coords)


def overlay_static_coords(norm_coords, img_file):
    fig,ax = plt.subplots(1)

    img_height, img_width = 2048, 4096
    coords = norm_coords * np.array([img_width, img_height])

    image = misc.imresize(misc.imread(img_file), (img_height, img_width))

    ax.imshow(image)
    ax.set_xlim([0,img_width])
    ax.set_ylim([img_height, 0])

    plt.scatter(coords[:,0], coords[:,1])
    plt.show(block=False)


def plot_heatmap(heatmap):
    fig,ax = plt.subplots(1)
    ax.imshow(heatmap, vmin=0.0, vmax=1.0, interpolation=None)
    plt.show(block=False)


def get_mode_pred_matrix(list_of_runs, columns=['gaze_lat_lon_offset', 'ang_head_velo', 'ang_head_acc', 'ang_gaze_velo', 'ang_gaze_acc']):
    all_columns_dict = {}
    for column in columns:
        array_list = []
        for run in list_of_runs:
            array_list.append(run[column])
        joined_array = np.vstack(array_list) #np.absolute(np.vstack(array_list))
        all_columns_dict[column] = joined_array

    data_matrix = np.column_stack(all_columns_dict.values())
    return data_matrix, all_columns_dict.keys()


def get_mode_classifier(data_matrix, column_keys):
    print("Computing k_means classifier.")
    feature_scaler = StandardScaler()
    feature_scaler.fit(data_matrix)
    norm_data_matrix = feature_scaler.transform(data_matrix)

    k_means = KMeans(n_clusters=2, n_init=20, verbose=True)
    predictions = k_means.fit_predict(norm_data_matrix)

    return k_means, feature_scaler


def visualize_combined_log(all_experiments_data,
                           img_file=os.path.join(DATASET_PATH, 'cubemap_0000.png'),
                           out_path=None,
                           show_fixations=True):
    max_length = max([len(experiment_data['timestamp']) for experiment_data in all_experiments_data])

    img_height, img_width = 2048, 4096

    fig,ax = plt.subplots(1, figsize=[24, 12])
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Display a time indicator
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', size='x-large', transform=ax.transAxes)

    image = misc.imresize(misc.imread(img_file), (img_height, img_width))

    ax.set_xlim([0,img_width])
    ax.set_ylim([img_height, 0])

    fov_deg = np.array([fov_vert, fov_vert*aspect_ratio])
    fov_pixels = fov_deg/np.array([180.0,360.0]) * np.array([img_height, img_width])

    # Create and add the rectangles and circles to the plot
    gaze_patches = []
    head_rects = []
    head_centers = []

    colors = cm.rainbow(np.linspace(0, 1, len(all_experiments_data)))
    for exp_data, color in zip(all_experiments_data, colors):
        gaze_patch = Circle((img_height/60, img_height/60), 25.0, fc=color, alpha=.5, edgecolor='none')
        head_rect = Rectangle((0,0), fov_pixels[1], fov_pixels[0], linewidth=2, fill=False, alpha=.4, edgecolor=color)
        head_center = Circle((img_height/160.0, img_height/160.0), 25.0, fc=color, alpha=.5, edgecolor='none')

        gaze_patches.append(gaze_patch)
        head_rects.append(head_rect)
        head_centers.append(head_center)


        if show_fixations:
            gaze_fix_coords = exp_data['gaze_fixations'] * np.array([img_width, img_height])
            ax.scatter(gaze_fix_coords[:, 0], gaze_fix_coords[:, 1], c=color)

    def init():
        img = ax.imshow(image, aspect='auto')
        ax.axis('tight')
        ax.grid(b=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        for head_rect, head_center, gaze_patch in zip(head_rects, head_centers, gaze_patches):
            ax.add_patch(gaze_patch)
            ax.add_patch(head_rect)
            # ax.add_patch(head_center)

        return gaze_patches + [img] + head_rects #+ head_centers + gaze_patches + [img]

    def update_gaze(i):
        time_text.set_text(time_template % (i * 1.0/120.0))

        for head_rect, head_center, gaze_patch, log_df in zip(head_rects, head_centers, gaze_patches, all_experiments_data):
            if i>= len(log_df['timestamp']): continue
            head_coords = lat_long_to_yx(log_df['headLatLon'][i, :], img_height, img_width)
            gaze_coords = lat_long_to_yx(log_df['gaze_lat_lon'][i, :], img_height, img_width)

            gaze_patch.center = gaze_coords[1], gaze_coords[0]
            head_rect.set_xy((head_coords[1]-0.5*fov_pixels[1], head_coords[0]-0.5*fov_pixels[0]))
            # head_center.center = head_coords[1], head_coords[0]

            if 'class' in log_df.keys():
                time_text.set_text(log_df['class'][i])
            # rotate transform
            # head_tilt = log_df['headTilt'][i]
            # tr = transforms.Affine2D().rotate_deg_around(gaze_lat_lon_centered_normalized[1], gaze_lat_lon_centered_normalized[0], head_tilt)
            # t = td2dis + tr

            # head_rect.set_transform(t)
            # if i%10 == 0:
            #     head_trail = Circle((head_coords[1], head_coords[0]), 25.0, fc=head_center.get_fc(), alpha=.3, edgecolor='none')
            #
            #     ax.add_patch(head_trail)
            #     head_trails.append(head_trail)
        return gaze_patches + [time_text] + head_rects
        # return head_rects + [time_text] + head_centers + gaze_patches

    print("starting animation")
    anim = animation.FuncAnimation(fig, update_gaze,
                                   frames=max_length,
                                   init_func=init,
                                   interval=1,
                                   blit=True)

    plt.show()

    if out_path:
        print("Saving animation to %s"%out_path)
        FFwriter = animation.FFMpegWriter(fps=100, bitrate=-1, codec="libx264")
        anim.save(out_path, writer=FFwriter)

    return anim


def write_camera_orientation_file(orientation_array, out_path):
    np.savetxt(X=orientation_array, fmt='%0.3f', fname=out_path, delimiter=",")


def overlay_image_heatmap(img_path, heatmap, plot=True, out_path=None):
    image = matplotlib.image.imread(img_path).astype(float)[:,:,:3]
    heatmap_resized = misc.imresize(heatmap, (image.shape[0], image.shape[1])).astype(float)

    fig,ax = plt.subplots(figsize=(16.0, 10.0))
    ax.imshow(image)
    plt.hold(True)
    ax.imshow(heatmap_resized, cmap=plt.cm.jet, alpha=0.4)

    if plot: show(block=False)
    if out_path:
        fig.savefig(out_path)#save_as_image(overlaid_image, out_path)
        plt.clf()

    return fig, ax


def head_long_heatmap(norm_xy_coords, plot=True, out_path=None):
    img_height, img_width = 256, 512
    coords = np.mod(norm_xy_coords,1.0) * np.array((img_width, img_height))
    coords = coords.astype(int)

    fov_deg = np.array([106, 106*0.89])
    fov_pixels = fov_deg/np.array([180.0,360.0]) * np.array([img_height, img_width])

    gaze_counts = np.zeros((img_height, img_width))

    for coord in coords:
        gaze_counts[:, coord[0]] += 1.0

    sigma = fov_pixels[1]/10.0
    heatmap = filters.gaussian_filter(gaze_counts, sigma=sigma, mode='wrap')
    heatmap /= np.maximum(0.01, np.amax(heatmap))

    if plot: plot_heatmap(heatmap)
    if out_path:
        plt.imsave(arr=heatmap, fname=out_path)

    return heatmap


def get_head_fixation_idcs(runs_list):
    all_fixation_idcs = []
    for curr_run in runs_list:
        temp_df = pd.DataFrame()
        run_length = len(curr_run['timestamp'])
        temp_df['index'] = np.arange(run_length)
        temp_df['timestamp'] = curr_run['timestamp']

        temp_df['norm_pos'] = lat_long_to_yx(curr_run['headLatLon'], 1.0, 1.0)[:, [1, 0]].tolist()
        temp_df['confidence'] = np.ones(run_length)

        temp_df['fixation'] = np.zeros(run_length)

        fix_dect = fixation_detector.Fixation_Detector_Dispersion_Duration(max_dispersion=15.0, min_duration=1, h_fov=360.0, v_fov=180.0)
        fixations = fix_dect.classify(temp_df.to_dict('records'))

        for fixation in fixations:
            temp_df.ix[fixation['start_frame_index']:fixation['end_frame_index'], 'fixation'] = 1

        all_fixation_idcs.append(np.where(temp_df['fixation'].values))
    return all_fixation_idcs



def debug_helper(np_arrays, idcs):
    relevant_parts = [np_array[idcs] for np_array in np_arrays]
    stacked = np.column_stack(relevant_parts)
    print(stacked)


def glob_all_image_files(dir):
    if dir[0]!='/':
        dir = os.path.join(EXEC_HOME, dir)
    image_paths = []
    filetypes = ['*.png', '*.jpg', '*.JPEG']
    for filetype in filetypes:
        image_paths.extend(glob.glob(dir + filetype))
    return image_paths


def stitch_gnomonic(gnomonic_dir, target_height_width, fov_vert_hor=(106.0, 96.0), plot=False):
    gnomonic_images = glob_all_image_files(gnomonic_dir)

    test_image = matplotlib.image.imread(gnomonic_images[0]).astype(float)
    shape = test_image.shape

    if len(shape) == 2:
        org_img_height, org_img_width = shape
        target_image = np.zeros((target_height_width[0], target_height_width[1]))
        num_channels=1
    else:
        org_img_height, org_img_width, num_channels = shape
        target_image = np.zeros((target_height_width[0], target_height_width[1], num_channels))

    fov_vert_hor = np.array(fov_vert_hor)

    target_pixel_counts = np.zeros(target_height_width)

    for image_path in gnomonic_images:
        patch_coords = np.array([int(num) for num in os.path.splitext(os.path.basename(image_path))[0].split('_')])
        origin_image = matplotlib.image.imread(image_path).astype(float)

        xx, yy = np.mgrid[0:org_img_width, 0:org_img_height].astype(float)
        x_y = np.column_stack([xx.ravel(), yy.ravel()])

        lat_lons = inv_projection_transformation(x_y_array=x_y,
                                                 height_width=(org_img_height, org_img_width),
                                                 fov_vert_hor=fov_vert_hor,
                                                 center_lat_lons=patch_coords.reshape(-1, 2))
        target_coords = np.mod(lat_lons, np.array([180, 360])) / np.array([180, 360]) * np.array(target_height_width)
        target_coords = target_coords.astype(int)
        target_pixel_counts[target_coords[:, 0], target_coords[:, 1]] += 1

        if num_channels>1:
            target_image[target_coords[:, 0], target_coords[:, 1], :num_channels] += origin_image[x_y[:, 1].astype(int), x_y[:, 0].astype(int), :num_channels]
        else:
            target_image[target_coords[:, 0], target_coords[:, 1]] += origin_image[x_y[:, 1].astype(int), x_y[:, 0].astype(int)]

    target_pixel_counts[target_pixel_counts==0] = 1.0

    if num_channels > 1:
        target_image /= target_pixel_counts.reshape((target_height_width[0], target_height_width[1], 1))
    else:
        target_image /= target_pixel_counts

    if plot:
        fig,ax = plt.subplots(figsize=(16.0, 10.0))
        if num_channels > 1:
            ax.imshow(target_image[:, :, :3])
        else:
            ax.imshow(target_image,  interpolation=None, cmap=plt.cm.jet)
            show(block=False)

    return target_image/np.amax(target_image)


def stitch_gnomonic_to_equirectangular(gnomonic_dir,
                                       target_height_width,
                                       fov_vert_hor=(106.0, 96.0),
                                       plot=False,
                                       normalize_first=True,
                                       patch_size=(480,480)):

    gnomonic_images = glob.glob(gnomonic_dir + '*')
    fov_vert_hor = np.array(fov_vert_hor)

    # Extract the center lat, lon coordinates of the gnomonic patches and preload all images
    patch_coords = np.zeros((len(gnomonic_images), 2))
    origin_images = np.zeros((len(gnomonic_images), patch_size[0], patch_size[1]))
    for idx, gnomonic_path in enumerate(gnomonic_images):
        filename_no_extension = os.path.basename(os.path.splitext(os.path.splitext(gnomonic_path)[0])[0])
        patch_coords[idx, :] = np.array(filename_no_extension.split('_'))

        with open(gnomonic_path, 'r') as pck_file:
            origin_images[idx] = pickle.load(pck_file)

        if normalize_first:
            origin_images[idx] /= np.amax(origin_images[idx])

    test_img = origin_images[0]
    num_channels = 1
    if len(test_img.shape) > 2:
        gnomonic_img = test_img[:, :, :3]
        origin_height, origin_width, num_channels = tuple(float(i) for i in gnomonic_img.shape)
        origin_height_width = np.array([origin_height, origin_width]).astype(float)
        target_image = np.zeros((target_height_width[0], target_height_width[1], num_channels))
    else:
        origin_height_width = np.array([float(i) for i in test_img.shape]).astype(float)
        target_image = np.zeros((target_height_width[0], target_height_width[1]))

    xx, yy = np.mgrid[0:target_height_width[1], 0:target_height_width[0]].astype(float)
    x_y = np.column_stack([xx.ravel(), yy.ravel()])

    sphere_radius_lon = origin_height_width[1] / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = origin_height_width[0]  / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    for x_pix, y_pix in x_y:
        # Find the closest patch
        curr_lat_lon = (np.stack([y_pix, x_pix])/target_height_width) * np.array([180.0, 360.0])
        best_origin_idx = np.argmin(np.linalg.norm(curr_lat_lon - patch_coords, axis=1))

        if (np.mod(x_pix, 100.0)==0 and not y_pix):
            print("x y %f %f, lat_lon %s, best fit %s"%(x_pix,y_pix, curr_lat_lon, gnomonic_images[best_origin_idx]))

        origin_image = origin_images[best_origin_idx]

        # Get the closest patch's center latitude and longitude
        origin_lat_lon = patch_coords[best_origin_idx, :].astype(float)

        # Get the world coordinates from latitude longitude of the unit sphere
        z_wrld = np.cos(np.radians(curr_lat_lon[0]))
        f = np.sin(np.radians(curr_lat_lon[0]))
        x_wrld = np.sin(np.radians(curr_lat_lon[1])) * f
        y_wrld = np.cos(np.radians(curr_lat_lon[1])) * f
        coords_wrld = np.array([x_wrld, y_wrld, z_wrld])

        # do the forward projection from world coordinates to the patch's viewport coordinates
        K = np.zeros((3, 3))
        K[0, 0] = sphere_radius_lon
        K[1, 1] = sphere_radius_lat
        K[0, 2] = origin_height_width[1] / 2.0
        K[1, 2] = origin_height_width[0] / 2.0
        K[2, 2] = 1.0

        R_lat = np.zeros((3, 3))
        R_lat[0, 0] = 1.0
        R_lat[1, 1] = np.cos(np.radians(-origin_lat_lon[0]))
        R_lat[2, 2] = R_lat[1, 1]
        R_lat[1, 2] = -1.0 * np.sin(np.radians(-origin_lat_lon[0]))
        R_lat[2, 1] = -1.0 * R_lat[1, 2]

        R_lon = np.zeros((3, 3))
        R_lon[2, 2] = 1.0
        R_lon[0, 0] = np.cos(np.radians(-origin_lat_lon[1]))
        R_lon[1, 1] = R_lon[0, 0]
        R_lon[0, 1] = - np.sin(np.radians(-origin_lat_lon[1]))
        R_lon[1, 0] = - R_lon[0, 1]

        R_full = np.matmul(R_lon, R_lat).T.astype(float)

        origin_x_y_z = np.sum(np.matmul(K, R_full) * coords_wrld.reshape(1, 3), axis=1)
        # print(coords_wrld)
        # print(curr_lat_lon)
        # print(origin_lat_lon)
        # origin_x_y_z = origin_x_y_z[:2]/origin_x_y_z[2]

        # x_gnomonic_dist = np.tan(np.radians(origin_lat_lon[1] - curr_lat_lon[1])) * sphere_radius_lon
        #
        # base_y = np.sqrt(
        #     np.power(sphere_radius_lat, 2) + np.power(x_gnomonic_dist * (sphere_radius_lat / sphere_radius_lon), 2))
        # y_gnomonic_dist = np.tan(np.radians(origin_lat_lon[0] - curr_lat_lon[0])) * base_y
        # gnomonic_dist = np.stack([x_gnomonic_dist, y_gnomonic_dist])
        # origin_x_y = np.clip(0.5 * origin_height_width - gnomonic_dist, 0.0, origin_height_width - 1.0)

        origin_x_y = np.clip(origin_x_y_z[:2], 0.0, origin_height_width-1.0).astype(int)
        origin_pixel = origin_image[origin_x_y[1], origin_x_y[0]]

        if num_channels > 1:
            target_image[int(y_pix), int(x_pix), :num_channels] = origin_pixel[:num_channels]
        else:
            target_image[int(y_pix), int(x_pix)] = origin_pixel

    target_image /= np.amax(target_image)

    if plot:
        fig,ax = plt.subplots(figsize=(16.0, 10.0))
        if num_channels > 1:
            ax.imshow(target_image)
        else:
            ax.imshow(target_image, vmin=0.0, vmax=1.0, interpolation=None, cmap=plt.cm.jet)
        show(block=False)

    return target_image


def extract_360_gnomonic(img_path, out_dir, fov_vert_hor=(106, 96), size=(128,128)):
    image = matplotlib.image.imread(img_path).astype(float)[:, :, :3]

    for y in np.arange(0.0, 180.0, np.floor(fov_vert_hor[0]/2.0)):
        for x in np.arange(0.0, 360.0, np.floor(fov_vert_hor[1]/2.0)):
            out_path = os.path.join(out_dir, "%d_%d.jpg"%(y, x))
            get_gnomonic_hom((y, x), image, size, fov_vert_hor=fov_vert_hor, out_path=out_path)


def inv_projection_transformation(x_y_array, height_width, center_lat_lons, fov_vert_hor):
    coords_hom = np.column_stack([x_y_array[:, 0], x_y_array[:, 1], np.ones(len(x_y_array))])

    sphere_radius_lon = height_width[1] / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = height_width[0]  / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0 / sphere_radius_lon
    K_inv[1, 1] = 1.0 / sphere_radius_lat
    K_inv[0, 2] = -height_width[1] / (2.0 * sphere_radius_lon)
    K_inv[1, 2] = -height_width[0] / (2.0 * sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((len(center_lat_lons), 3, 3))
    R_lat[:, 0, 0] = 1.0
    R_lat[:, 1, 1] = np.cos(np.radians(-center_lat_lons[:, 0]))
    R_lat[:, 2, 2] = R_lat[:, 1, 1]
    R_lat[:, 1, 2] = -1.0 * np.sin(np.radians(-center_lat_lons[:, 0]))
    R_lat[:, 2, 1] = -1.0 * R_lat[:, 1, 2]

    R_lon = np.zeros((len(center_lat_lons), 3, 3))
    R_lon[:, 2, 2] = 1.0
    R_lon[:, 0, 0] = np.cos(np.radians(-center_lat_lons[:, 1]))
    R_lon[:, 1, 1] = R_lon[:, 0, 0]
    R_lon[:, 0, 1] = - np.sin(np.radians(-center_lat_lons[:, 1]))
    R_lon[:, 1, 0] = - R_lon[:, 0, 1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(-1, 3, 3) * coords_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod / np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    return lat_lon


def get_gnomonic_hom(center_lat_lon, origin_image, height_width, fov_vert_hor=(60.0, 60.0), out_path=None,
                     plot=False):
    org_height_width = origin_image.shape[:2]
    single_channel = len(origin_image.shape)==2

    height, width = height_width

    if single_channel:
        result_image = np.zeros((height, width))
    else:
        result_image = np.zeros((height, width, 3))

    sphere_radius_lon = width / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = height / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    y, x = np.mgrid[0:height, 0:width]
    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0/sphere_radius_lon
    K_inv[1, 1] = 1.0/sphere_radius_lat
    K_inv[0, 2] = -width/(2.0*sphere_radius_lon)
    K_inv[1, 2] = -height/(2.0*sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3,3))
    R_lat[0,0] = 1.0
    R_lat[1,1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2,2] = R_lat[1,1]
    R_lat[1,2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2,1] = -1.0 * R_lat[1,2]

    R_lon = np.zeros((3,3))
    R_lon[2,2] = 1.0
    R_lon[0,0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1,1] = R_lon[0,0]
    R_lon[0,1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1,0] = - R_lon[0,1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1,3,3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod/np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    org_img_y_x = lat_lon / np.array([180.0, 360.0]) * np.array(org_height_width)
    org_img_y_x = np.clip(org_img_y_x, 0.0, np.array(org_height_width).reshape(1, 2) - 1.0).astype(int)
    org_img_y_x = org_img_y_x.astype(int)

    if not single_channel:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int), :] = origin_image[org_img_y_x[:, 0],
                                                                        org_img_y_x[:, 1], :]
    else:
        result_image[x_y_hom[:, 1].astype(int), x_y_hom[:, 0].astype(int)] = origin_image[org_img_y_x[:, 0],
                                                                                org_img_y_x[:, 1]]

    if plot:
        fig, ax = plt.subplots(figsize=(16.0, 10.0))
        ax.imshow(result_image.astype('uint8'))
        show(block=False)

    if out_path: plt.imsave(out_path, result_image)

    return result_image, org_img_y_x


def gaussian_blur_gaze_counts(gaze_counts):
    sigma = gaze_counts.shape[1]/360.0
    heatmap = filters.gaussian_filter(gaze_counts, sigma=sigma, mode='wrap')
    heatmap /= np.maximum(0.01, np.amax(heatmap))
    return heatmap


def get_random_heatmap(height_width):
    result = np.random.uniform(low=0.0, high=100.0, size=height_width)
    result /= np.amax(result)
    return result


def pearsons_heatmap_cc(heatmap1, heatmap2):
    score, p_value = scipy.stats.pearsonr(heatmap1.reshape(-1), heatmap2.reshape(-1))
    return score


def get_angular_speed_per_region(list_of_runs, img_file, plot=False, out_path=None):
    runs_with_speed = []
    for run in list_of_runs:
        runs_with_speed.append(preprocessing.get_closest_timestamp_merge(run, get_angular_speed(run, False), []))

    angular_bins = np.arange(0, 360, 5)
    average_speeds = np.zeros(len(angular_bins))
    for run in runs_with_speed:
        for i in xrange(1, len(angular_bins)):
            relevant_idcs = np.where( (run['headLatLon'][:, 1] < angular_bins[i]) & (run['headLatLon'][:,1] > angular_bins[i-1]))
            relevant_speeds = run['speed'][relevant_idcs, 1]
            average_speeds[i-1] = np.mean(np.absolute(relevant_speeds))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(32.0, 18.0))
    ax1.bar((angular_bins[:-1] + angular_bins[1:]) / 2, average_speeds[:-1], width=5)
    ax1.set_xlim([0, 360])

    img =misc.imread(img_file)
    ax2.imshow(img)

    if plot: show(block=False)
    if out_path:
        f.savefig(out_path)
        plt.clf()


def get_left_right_turners(list_of_runs, viewpoint_idx):
    left_turners = []
    right_turners = []
    left_border = (viewpoint_idx * 90.0 - 30) % 360
    right_border = (viewpoint_idx * 90.0 + 30) % 360

    for run in list_of_runs:
        headLon = run['headLatLon'][:,1]

        left_border_touches = np.where(np.logical_and(headLon<left_border+10.0, headLon>left_border-10.0))[0]
        right_border_touches = np.where(np.logical_and(headLon<right_border+10.0, headLon>right_border-10.0))[0]

        if len(left_border_touches) != 0 and len(right_border_touches) != 0:
            left_border_time = run['timestamp'][np.amin(left_border_touches)]
            right_border_time = run['timestamp'][np.amin(right_border_touches)]

            if left_border_time or  right_border_time:
                if left_border_time < right_border_time:
                    left_turners.append(run)
                else:
                    right_turners.append(run)
        elif len(left_border_touches) != 0 or len(right_border_touches) != 0:
            if len(left_border_touches) != 0:
                left_turners.append(run)
            else:
                right_turners.append(run)
        else:
            print("Nothing at all!")

    return left_turners, right_turners


def get_horizon_bias_heatmap(height_width=(256, 512), path=DATASET_PATH, type='counts', mode='single'):
    print("Calculating the horizon bias heatmap with %s standard"%type)
    all_runs = load_all_runs(path=path)
    # Filter the starting points
    filtered_list = filter_starting_point(all_runs)

    fixation_norm_yx = get_fixation_norm_xy(filtered_list, mode=mode)[:, [1, 0]]

    average_heatmap = heatmap_from_norm_yx(fixation_norm_yx, height_width=height_width, type=type)
    final_heatmap = np.zeros_like(average_heatmap)
    final_heatmap[:, :] = np.mean(average_heatmap, axis=1, keepdims=True)
    final_heatmap /= np.amax(final_heatmap)
    print("finished horizon bias")

    return average_heatmap / np.sum(average_heatmap)#final_heatmap


def get_average_heatmap(list_of_runs, height_width=(256, 512), type='counts'):
    fixation_norm_yx = get_fixation_norm_xy(list_of_runs)[:, [1, 0]]
    average_heatmap = heatmap_from_norm_yx(fixation_norm_yx, height_width=height_width, type=type)

    return average_heatmap / np.sum(average_heatmap)


def heatmap_from_norm_yx(norm_coords, height_width=(256, 512), type='counts', sigma=None):
    img_height, img_width = height_width
    img_coords = np.mod(np.round(norm_coords * np.array(height_width)), np.array(height_width)-1.0).astype(int)

    gaze_counts = np.zeros((img_height, img_width))
    for coord in img_coords:
        if type == 'counts':
            gaze_counts[coord[0], coord[1]] += 1.0
        elif type == 'binary':
            gaze_counts[coord[0], coord[1]] = 1.0

    gaze_counts[0, 0] = 0.0

    if sigma is None:
        sigma = img_width/360.0

    heatmap = filters.gaussian_filter(gaze_counts, sigma=sigma, mode='wrap')
    heatmap /= float(np.sum(heatmap))

    return heatmap


def get_fixation_norm_xy(list_of_runs, mode='all', time=None):
    fixation_coords = []

    if mode=='all':
        column = 'gaze_fixations'
    elif mode=='single':
        column = 'gaze_fixations_single'

    for run in list_of_runs:
        if time:
            correct_times = run['timestamp'] < time
            relevant_fixations = run[column][correct_times]
        else:
            relevant_fixations = run[column]

        if len(relevant_fixations.shape) > 1:
            _, unique_idcs = np.unique(relevant_fixations[:, 0], return_index=True)
            all_fixations = relevant_fixations[unique_idcs]
            fixation_coords.append(all_fixations)

    return np.vstack(fixation_coords)


def get_gaze_counts(list_of_runs, time=None, height_width=(720, 1440)):
    fixation_norm_yx = get_fixation_norm_xy(list_of_runs, time=time)[:, [1, 0]]

    img_coords = np.mod(np.round(fixation_norm_yx * np.array(height_width)), np.array(height_width)-1.0).astype(int)

    gaze_counts = np.zeros(height_width)
    for coord in img_coords:
        gaze_counts[coord[0], coord[1]] += 1.0

    gaze_counts[0, 0] = 0.0

    return gaze_counts


def get_gaze_counts_fast(list_of_runs, time=None, height_width=(720, 1440)):
    img_height, img_width = height_width

    fixation_norm_yx = get_fixation_norm_xy(list_of_runs, time=time)[:, [1, 0]]

    coords = fixation_norm_yx * np.array((img_height, img_width))
    coords = coords.astype(int)

    gaze_counts = np.zeros((img_height, img_width))
    for coord in coords:
        gaze_counts[coord[0], coord[1]] += 1.0

    gaze_counts[0, 0] = 0.0

    return gaze_counts


def load_all_browser_runs():
    print(os.path.join(DATASET_PATH_BROWSER, 'all_runs_browser.pck'))
    all_runs = []
    for i in xrange(22):
        loaded_logfile = load_logfile(os.path.join(DATASET_PATH_BROWSER, "%d.pck"%i))['data']
        all_runs.extend(loaded_logfile)

    with open(os.path.join(DATASET_PATH_BROWSER, 'all_runs_browser.pck'), 'w') as pck_file:
        msg.dump(all_runs, pck_file)



def get_gaze_heatmap(list_of_runs, time=None, height_width=(720, 1440), sigma=None, mode='single'):
    # Get this cubemap's normalized gaze fixation coordinates
    fixation_norm_yx = get_fixation_norm_xy(list_of_runs, mode=mode, time=time)[:, [1, 0]]

    # Get this cubemap's gaze heatmap
    heatmap = heatmap_from_norm_yx(fixation_norm_yx, height_width=height_width, sigma=sigma, type='counts')

    return heatmap


def get_head_heatmap(list_of_runs, thresh=37.196, height_width=(720, 1440), sigma=None):
    # Get this cubemap's normalized gaze fixation coordinates
    all_head_velos = []
    all_head_lat_lons = []

    for run in list_of_runs:
        all_head_velos.append(run['ang_head_velo'])
        all_head_lat_lons.append(run['headLatLon'])

    head_velos = np.vstack(all_head_velos)
    head_lat_lons = np.vstack(all_head_lat_lons)

    fixation_idcs = head_velos[:, 1] < thresh
    fix_lat_lons = head_lat_lons[fixation_idcs]
    norm_fix_coords = fix_lat_lons / np.array([180, 360])

    # Get this cubemap's gaze heatmap
    heatmap = heatmap_from_norm_yx(norm_fix_coords, height_width=height_width, sigma=sigma)

    return heatmap



def cubemap_iterator(path=DATASET_PATH, subset=[]):
    for i in xrange(22):
        if subset and i not in subset:
            continue
        logfile = os.path.join(path, 'cubemap_00%02d.pck'%i)
        img_file = os.path.join(path, 'cubemap_00%02d.png'%i)
        yield logfile, img_file


def load_all_runs(path=DATASET_PATH):
    if path.endswith('pck'):
        print("\n*****WARNING****** \nLoading buffered list of all runs from %s."%path)
        with open(path, 'r') as pickle_file:
            return msg.load(pickle_file)

    if os.path.exists(os.path.join(path, 'all_runs_list.pck')):
        print("\n*****WARNING****** \nLoading buffered list of all runs.")
        with open(os.path.join(path, 'all_runs_list.pck'), 'r') as pickle_file:
            return msg.load(pickle_file)

    all_runs = []
    for i in xrange(22):
        logfile = os.path.join(path, 'cubemap_00%02d.pck'%i)
        print(logfile)
        all_runs.extend(load_logfile(logfile)['data'].tolist())

    with open(os.path.join(path, 'all_runs_list.pck'), 'w') as pickle_file:
        msg.dump(all_runs, pickle_file)
    return all_runs


def main():
    img_file = 'data/cubemap_0000.png'
    logfile = 'data/cubemap_0000.pck'

    all_runs = load_all_runs()
    # data_matrix, column_names = get_mode_pred_matrix(all_runs, columns=['gaze_lat_lon_offset', 'ang_head_velo', 'ang_gaze_velo'])
    # classifier, scaler = get_mode_classifier(data_matrix, column_names)
    # data_matrix_with_class = np.column_stack([data_matrix, classifier.predict(scaler.transform(data_matrix))])
    #
    # get_pandas_overview(data_matrix_with_class, column_names+'class')

    # nrows = int(np.ceil(len(column_names) / 2.))
    # fig, axs = plt.subplots(nrows, 2)
    #
    # for i in xrange(nrows):
    #     for j in xrange(2):
    #         if i*2 + j == data_matrix.shape[1]: break
    #         n, bins, patches = axs[i, j].hist(data_matrix[i*2 + j], 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()


    # test_run = all_runs[:3]

    # mode_classifier, standardizer = get_mode_classifier(all_runs[:-1])
    # test_matrix, column_keys = get_mode_pred_matrix([test_run])
    # test_run['mode'] = mode_classifier.predict(standardizer.transform(test_matrix))
    # print(test_run['mode'])

    # visualize_combined_log(test_run, img_file=img_file)


    # print("All runs stats")
    # get_stats(all_runs['data'].tolist())
    # print("\n")
    #
    # all_runs = all_runs[all_runs['viewpoint_idx']==2]['data']
    # left_turners, right_turners = get_left_right_turners(all_runs, 2)
    # print("Left turner stats")
    # get_stats(left_turners)
    # print("\n")
    # combined_heatmap(left_turners)


    # all_runs_df = load_logfile(logfile)
    # get_head_angular_speeds(all_runs_df)
    # print(all_runs_df.iloc[0]['data']['speed'])
    # get_head_fixation_idcs(all_runs_df)
    # add_xy_norm_position_column(all_runs_df, ['headLatLon'])
    # joined_df = pd.concat(all_runs_df['data'].tolist())
    # overlay_static_coords(np.vstack(joined_df[joined_df['head_fixation']==1.0]['norm_pos']), img_file)

    # viewpoint_0 = all_logs_df[all_logs_df['viewpoint_idx']==3]['data'].tolist()
    #
    # left_turners, right_turners = get_left_right_turners(viewpoint_0, 3)
    #
    # visualize_combined_log(left_turners, img_file)

    # head_fixation_idcs = get_angular_speed(first_run, plot=True)
    # fixations = lat_long_to_img_coords(first_run['headLatLon'][head_fixation_idcs[0]], 1.0, 1.0)[:, [1,0]]
    #
    # visualize_combined_log([first_run], img_file, fixations)
    show()


def inf_gain(baseline, prediction, ground_truth_fixations, epsilon=1e-20):
    baseline /= np.sum(baseline)
    prediction /= np.sum(prediction)
    ground_truth_fixations = (ground_truth_fixations > 0).astype(float)

    N_inv = 1.0 / np.sum(ground_truth_fixations)

    return N_inv * np.sum(ground_truth_fixations * (np.log2(1e-20 + prediction) - np.log2(1e-20 + baseline)))

def inf_gain_distrib(baseline, ground_truth, prediction, epsilon=1e-20):
    baseline /= np.sum(baseline)
    ground_truth /= np.sum(ground_truth)
    prediction /= np.sum(prediction)

    N_inv = 1.0 / np.prod(ground_truth.shape)

    return N_inv * np.sum((np.log2(1e-20 + ground_truth) - np.log2(1e-20 + baseline)))

if __name__=='__main__':
    main()
