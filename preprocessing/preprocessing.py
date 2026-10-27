from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from scipy import interpolate
from scipy import io

import copy
import cPickle as pickle
import os

import settings
import fixation_detector

import data_analysis

plt.style.use('ggplot')
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Demo Account\\Documents\\ffmpeg\\bin\\ffmpeg.exe'

np.set_printoptions(threshold=np.inf)
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def filter_wrong_timestamps(log):
    iterator = log.iterrows()
    comparison_counter = 0
    while iterator.next()[1]['timestamp'] > 1.0:
        comparison_counter += 1

    return log.iloc[comparison_counter+1:]


def interpolate_nan_rows(array, bad_rows_bool):
    if True not in bad_rows_bool.astype(bool): return array

    good_rows = np.where(np.logical_not(bad_rows_bool))[0]

    # Since extrapolation is a bad idea, we identify the low-confidence indices that are outside the interpolatable range.
    low_non_nan, high_non_nan = np.amin(good_rows), np.amax(good_rows)
    interp_idcs = np.copy(bad_rows_bool)
    interp_idcs[high_non_nan:] = False
    interp_idcs[:low_non_nan] = False

    extra_idcs = np.logical_and(bad_rows_bool,
                                np.logical_or(np.arange(len(bad_rows_bool)) <= low_non_nan,
                                              np.arange(len(bad_rows_bool)) >= high_non_nan))

    interp_func = interpolate.interp1d(good_rows,
                                       np.take(array, good_rows, axis=0),
                                       kind='linear',
                                       axis=0)
    array[interp_idcs] = interp_func(np.where(interp_idcs)[0])
    # The indices outside the interpolatable range are set to the mean of the series.
    array[extra_idcs] = np.mean(array[good_rows], axis=0)

    # interp_func = interpolate.interp1d(good_rows,
    #                                    np.take(array, good_rows, axis=0),
    #                                    kind='nearest',
    #                                    fill_value='extrapolate',
    #                                    axis=0)

    #
    # nan_idcs = np.where(bad_rows_bool)[0]
    # array[nan_idcs] = interp_func(nan_idcs)

    return array


def get_ang_va(run, column, filter_wrappers=True):
    ang_velo = np.zeros_like(run[column])
    ang_acc = np.zeros_like(run[column])

    ang_diff = np.diff(run[column], axis=0)
    time_diff = np.squeeze(np.diff(run['timestamp'], axis=0))

    if filter_wrappers:
        # Filter bad measurements due to wrapping or measurement errors.
        # We find these measurements by finding measurements where the difference is higher than 25 (this is akin to
        # filtering speeds of more than 3000 degree per second (for 120 hz) or 1500 degree per second (for 60 hz)
        # which is approximately twice as high as peak gaze/head speeds (1000 and 800 degrees per second)) and ensuring
        # that this occured due to a wrap-around (jumping from >179/359 (lat/lon) to less than 1 degree)
        bad_diff = np.logical_and(run[column][:, 1]>170.0,
                                  np.logical_or(np.roll(run[column][:, 1]<45.0, 1),
                                                np.roll(run[column][:, 1]<45.0, -1)))
        #
        # bad_diff_lat = np.logical_or(run[column][:, 0]>150.0,
        #                           np.logical_or(np.roll(run[column][:, 0]<10.0, 1),
        #                                         np.roll(run[column][:, 0]<10.0, -1)))

        # bad_diff = np.logical_or(bad_diff_lat, bad_diff_lon)
        bad_diff = np.logical_or(bad_diff, np.roll(bad_diff, 1))
        bad_diff = np.logical_or(bad_diff, np.roll(bad_diff, -1))
        bad_diff = np.roll(bad_diff, -1)[:-1]

        # Measurement errors are interpolated.
        ang_diff = interpolate_nan_rows(ang_diff, bad_diff)
        print('Interpolated %f percent of speed measurements'%(float(np.sum(bad_diff))/len(run[column])))

    ang_velo[:-1] = ang_diff / time_diff.reshape([-1, 1])

    ang_v_diff = np.diff(ang_velo, axis=0)
    ang_acc[:-1] = ang_v_diff / time_diff.reshape([-1,1])

    return ang_velo, ang_acc


def parse_logfile(log_filepath, prototype_row):
    def _to_numpy_vector(cs_vector_string):
        return np.array(cs_vector_string[1:-1].replace(', ', ',').split(',')).astype(float)

    data = []
    column_keys = []
    with open(log_filepath, 'r') as log_file:
        for idx, line in enumerate(log_file):
            if idx == 0: continue

            single_members = line.strip().split(';')
            if idx == 1:
                column_keys = single_members
            else:
                single_datapoint = []
                for column, single_member in zip(column_keys, single_members):
                    if single_member[0] == '(':
                        processed_member = _to_numpy_vector(single_member)
                    else:
                        try:
                            processed_member = float(single_member)
                        except ValueError:
                            processed_member =  single_member

                    single_datapoint.append(processed_member)

                correct_shape = True
                for prototype, candidate in zip(prototype_row, single_datapoint):
                    if type(prototype)==np.ndarray:
                        if not prototype.shape == candidate.shape:
                            correct_shape=False

                if correct_shape: data.append(single_datapoint)

    data_dict = {}
    for idx, column in enumerate(column_keys):
        data_dict[column] = np.vstack([single_datapoint[idx] for single_datapoint in data])

    return data_dict


def get_closest_timestamp_merge(dict_1, dict_2, mode='max', suffixes=[]):
    def _get_closest_timestamp_merge(dict_1, dict_2, suffixes):
        timestamps_1 = dict_1['timestamp']
        timestamps_2 = dict_2['timestamp']

        distances = np.absolute(timestamps_2.reshape([1, -1]) - timestamps_1.reshape([-1, 1]))
        closest_t2 = np.argmin(distances, axis=1)
        closest_t1 = np.argmin(distances, axis=0).reshape(-1)

        if suffixes:
            dict_1 = {(key+suffixes[0] if key != 'timestamp' else key):item for key, item in dict_1.iteritems()}
            dict_2 = {(key+suffixes[1] if key != 'timestamp' else key):item for key, item in dict_2.iteritems()}

        if mode=='max':
            joined_dict = dict_1.copy()
            for key, item in dict_2.iteritems():
                if key == 'timestamp': continue
                joined_dict.update({key:dict_2[key][closest_t2]})
        elif mode=='min':
            joined_dict = dict_2.copy()
            for key, item in dict_1.iteritems():
                if key == 'timestamp': continue
                joined_dict.update({key:dict_1[key][closest_t1]})

        return joined_dict

    if len(dict_1['timestamp']) > len(dict_2['timestamp']):
        return _get_closest_timestamp_merge(dict_1, dict_2, suffixes)
    else:
        return _get_closest_timestamp_merge(dict_2, dict_1, suffixes[::-1])


def dict_rows_subset(target_dict, row_idcs):
    target_dict_copy = copy.deepcopy(target_dict)

    for key, item in target_dict.iteritems():
        if key ==  'gaze_fixations_raw':
            continue

        try:
            target_dict_copy[key] = item[row_idcs]
        except:
            print("error in key %s. Lengths: %d and %d"%(key, len(target_dict_copy[key]), np.amax(row_idcs)))

    return target_dict_copy


def join_left_right_gaze(gaze_dict):
    '''
    Merges left and right eye records and averages them (that's how the eyetracker was calibrated).
    '''
    left_idcs = np.where(gaze_dict['eye'] == 'left')[0]
    right_idcs = np.where(gaze_dict['eye'] == 'right')[0]

    left_gaze_dict = {key:np.squeeze(np.take(item, left_idcs, axis=0)) for key, item in gaze_dict.iteritems()}
    right_gaze_dict = {key:np.squeeze(np.take(item, right_idcs, axis=0)) for key, item in gaze_dict.iteritems()}

    left_gaze_dict.pop('eye')
    right_gaze_dict.pop('eye')

    both_gaze_dict = get_closest_timestamp_merge(left_gaze_dict, right_gaze_dict, mode='max', suffixes=['_left', '_right'])
    both_gaze_dict['both_coords'] = (both_gaze_dict['coords_right'] + both_gaze_dict['coords_left'])/2.0
    both_gaze_dict['both_confidence'] = np.mean(np.stack([both_gaze_dict['confidence_left'],both_gaze_dict['confidence_right']]), axis=0)

    return both_gaze_dict


def clean_gaze_records(gaze_dict, confidence_threshold):
    non_faulty_idcs = np.ones(len(gaze_dict['timestamp']))

    # Remove the first few records with timestamps before the timestamps were reset
    for idx, timestamp in enumerate(gaze_dict['timestamp']):
        if timestamp<1.0: break
        non_faulty_idcs[idx] = False

    clean_beginning = dict_rows_subset(gaze_dict, np.where(non_faulty_idcs))
    print("Deleted %d incorrect timesteps"%(len(clean_beginning['timestamp']) - len(gaze_dict['timestamp'])))

    # Set the confidence of offsets that are larger than 1.5 / -0.5 to zero.
    bad_idcs = np.any(np.logical_or(clean_beginning['both_coords']>1.5, clean_beginning['both_coords']<-0.5), axis=1)
    clean_beginning['both_confidence'][bad_idcs] = 0.0

    # Interpolate the gaze positions with a low confidence
    low_conf_bool = clean_beginning['both_confidence']<confidence_threshold
    high_conf_bool = clean_beginning['both_confidence']>confidence_threshold

    try:
        # Since extrapolation is a bad idea, we identify the low-confidence indices that are outside the interpolatable range.
        low_non_nan, high_non_nan = np.amin(np.where(high_conf_bool)[0]), np.amax(np.where(high_conf_bool)[0])
        interp_idcs = np.copy(low_conf_bool)
        interp_idcs[high_non_nan:] = False
        interp_idcs[:low_non_nan] = False

        extra_idcs = np.logical_and(low_conf_bool,
                                    np.logical_or(np.arange(len(low_conf_bool))<=low_non_nan, np.arange(len(low_conf_bool))>=high_non_nan))

        interp_func = interpolate.interp1d(np.where(high_conf_bool)[0],
                                           np.take(clean_beginning['both_coords'], np.where(high_conf_bool)[0], axis=0),
                                           kind='nearest',
                                           axis=0)
        clean_beginning['both_coords'][interp_idcs] = interp_func(np.where(interp_idcs)[0])
        # The indices outside the interpolatable range are set to the mean of the series.
        clean_beginning['both_coords'][extra_idcs] = np.mean(clean_beginning['both_coords'][high_conf_bool], axis=0)
    except:
        print(clean_beginning['both_coords'][:100])
        print(clean_beginning['both_confidence'][:100])
        print(low_conf_bool[:100])

    print("Interpolated %f percent of measurements for low confidence"%(float(np.sum(low_conf_bool))/len(gaze_dict['timestamp'])))

    return clean_beginning


def get_starting_viewpoint(head_dict):
    viewpoint_longitude_clusters = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    cluster_distances = np.abs(viewpoint_longitude_clusters - (head_dict['headLatLon'][0, 1])%360.0)
    closest_cluster = np.argmin(cluster_distances)

    if closest_cluster == 4:
        closest_cluster = 0
    return closest_cluster


def gaze_account_head_tilt(gaze_coords, head_tilts):
    print("Accounting for head tilt...")

    theta = np.radians(head_tilts)
    c, s = np.cos(theta), np.sin(theta)
    first_row = np.column_stack([c, -1.0 * s])
    second_row = np.column_stack([s, c])
    matrix_array = np.stack([first_row, second_row], axis=-1).T

    gaze_coords_transposed_centered = gaze_coords.T.reshape([1, 2, -1]) - 0.5
    broadcasted_matrix_vector_product = gaze_coords_transposed_centered * matrix_array
    gaze_coords_rotated = np.sum(broadcasted_matrix_vector_product, axis=1).T + 0.5

    return gaze_coords_rotated



def get_gaze_lat_lon_offset(gaze_coords, head_tilt):
    print("Calculating angular gaze offset...")

    gaze_coords_rotated = gaze_account_head_tilt(gaze_coords, head_tilts=head_tilt)
    gaze_coords_centered = gaze_coords_rotated - 0.5

    ys, xs = gaze_coords_centered[:, 1], gaze_coords_centered[:, 0]
    # Pupil labs screen coordinate origin is at the bottom left. Since our latitude is +90 at the north pole, this aligns
    # with our coordinates just fine.
    lat_offset = np.arctan2(2.6625717 * ys, 0.9997617)
    lon_offset = np.arctan2(2.6625717 * 0.82034051 * xs, 0.9997617)

    result = np.degrees(np.column_stack([lat_offset, lon_offset]))
    result = np.nan_to_num(result)

    return np.array(result.tolist())


def get_gaze_lat_lon(gaze_coords, center_lat_lon, head_tilt):
    print("Calculating gaze latitude, longitude...")

    gaze_coords_rotated = gaze_account_head_tilt(gaze_coords, head_tilts=head_tilt)

    ys, xs = 1 - gaze_coords_rotated[:, 1], gaze_coords_rotated[:, 0]

    rotated_coords_hom = np.column_stack([xs, ys, np.ones(len(xs))])

    # Parameters of the field of view that unity sets when using the DK2.
    fov_v = 106.18877
    fov_h = 0.82034051 * fov_v

    sphere_radius_lon = 1.0 / (2.0 * np.tan(np.radians(fov_h / 2.0)))
    sphere_radius_lat = 1.0 / (2.0 * np.tan(np.radians(fov_v / 2.0)))

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0 / sphere_radius_lon
    K_inv[1, 1] = 1.0 / sphere_radius_lat
    K_inv[0, 2] = -1.0 / (2.0 * sphere_radius_lon)
    K_inv[1, 2] = -1.0 / (2.0 * sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((len(center_lat_lon), 3, 3))
    R_lat[:, 0, 0] = 1.0
    R_lat[:, 1, 1] = np.cos(np.radians(-center_lat_lon[:, 0]))
    R_lat[:, 2, 2] = R_lat[:, 1, 1]
    R_lat[:, 1, 2] = -1.0 * np.sin(np.radians(-center_lat_lon[:, 0]))
    R_lat[:, 2, 1] = -1.0 * R_lat[:, 1, 2]

    R_lon = np.zeros((len(center_lat_lon), 3, 3))
    R_lon[:, 2, 2] = 1.0
    R_lon[:, 0, 0] = np.cos(np.radians(-center_lat_lon[:, 1]))
    R_lon[:, 1, 1] = R_lon[:, 0, 0]
    R_lon[:, 0, 1] = - np.sin(np.radians(-center_lat_lon[:, 1]))
    R_lon[:, 1, 0] = - R_lon[:, 0, 1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(-1, 3, 3) * rotated_coords_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod / np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    return lat_lon


def parse_head_logfile(logfile_path):
    protoype_row = [0.0, np.zeros((4)), np.zeros((3)), np.zeros((3)), np.zeros((2)), 0.0, np.zeros((2))]
    return parse_logfile(logfile_path, protoype_row)


def parse_gaze_logfile(logfile_path):
    protoype_row = [0.0, 'string', np.zeros((2)), 0.0]
    return parse_logfile(logfile_path, protoype_row)


def get_eye_fixations(log, mode='single'):
    print("Getting eye fixations...")
    temp_df = pd.DataFrame()
    temp_df['index'] = np.arange(len(log['timestamp']))
    temp_df['timestamp'] = log['timestamp']

    temp_df['norm_pos'] = data_analysis.lat_long_to_yx(log['gaze_lat_lon'], 1.0, 1.0)[:, [1, 0]].tolist()
    temp_df['confidence'] = log['both_confidence']

    temp_df['fixation'] = np.zeros(len(log['timestamp']))
    curr_run_dict = temp_df.to_dict('records')

    fix_dect = fixation_detector.Fixation_Detector_Dispersion_Duration()
    fixations_dict = fix_dect.classify(curr_run_dict)

    fixations_all = np.zeros((len(log['timestamp']), 2))
    fixations_single = np.zeros_like(fixations_all)
    for fixation in fixations_dict:
            fixations_single[(fixation['start_frame_index'] + fixation['end_frame_index'])/2] = fixation['norm_pos']
            fixations_all[fixation['start_frame_index']:fixation['end_frame_index']] = fixation['norm_pos']

    fixations_indicator = (fixations_all[:, 0] !=0 ).astype(int)

    return fixations_all, fixations_single, fixations_indicator, fixations_dict


def consolidate_log_to_dfs(experiment_dir,
                           out_dir,
                           fix_mode='single',
                           save_mat=False,
                           mean_conf_thresh=0.8,
                           inspect=True):

    dir_list = [name for name in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, name))]

    total_runs = 0
    num_discarded = 0

    for cubemap_dir in dir_list:
        # Glob and sort all logfiles for both gaze and head orientation
        gaze_logfiles = []
        head_logfiles = []

        for name in os.listdir(os.path.join(experiment_dir, cubemap_dir)):
            if 'csv' in name:
                complete_path = os.path.join(experiment_dir, cubemap_dir, name)
                if 'gaze' in name:
                    gaze_logfiles.append(complete_path)
                elif 'head' in name:
                    head_logfiles.append(complete_path)

        gaze_logfiles = sorted(gaze_logfiles)
        head_logfiles = sorted(head_logfiles)

        all_runs = []
        try:
            os.mkdir(os.path.join(settings.DATASET_PATH, cubemap_dir))
        except:
            pass

        # Loop through head, gaze logfile pairs
        for head_logfile, gaze_logfile in zip(head_logfiles, gaze_logfiles):
            total_runs += 1

            head_dict = parse_head_logfile(head_logfile)
            gaze_dict = parse_gaze_logfile(gaze_logfile)

            print("Preprocessing file %s"%head_logfile)

            experiment_id = os.path.basename(head_logfile).split('_')[2]

            # join the left and right eye data. They are averaged to account for the both-eye-calibration.
            full_gaze_dict = join_left_right_gaze(gaze_dict)
            # Clean the gaze dataset, throwing away low-confidence data, wrong timestamps and demonstrable wrong measurements
            clean_gaze_dict = clean_gaze_records(full_gaze_dict, confidence_threshold=0.8)

            # If the average confidence of the gaze measurements is too low, throw the whole measurement away:
            if np.mean(clean_gaze_dict['both_confidence'])<mean_conf_thresh:
                print('\n')
                print(''.join(['*' for _ in xrange(100)]))
                print("Measurement %s is faulty, has mean confidence of %0.2f. Skipping."
                      %(head_logfile, np.mean(clean_gaze_dict['both_confidence'])))
                print(''.join(['*' for _ in xrange(100)]))
                print('\n')
                num_discarded += 1
                continue

            if inspect:
                fig, ax = plt.subplots(2)
                ax[0].scatter(full_gaze_dict['both_coords'][:, 1], full_gaze_dict['both_coords'][:, 0])
                ax[1].scatter(clean_gaze_dict['both_coords'][:, 1], clean_gaze_dict['both_coords'][:, 0])
                plt.show()

            # Limit the length of all records to exactly 30 seconds
            # clean_gaze_dict = dict_rows_subset(clean_gaze_dict, np.where(clean_gaze_dict['timestamp']<=30.0)[0])
            # head_dict = dict_rows_subset(head_dict, np.where(head_dict['timestamp']<=30.0)[0])

            # Calculate the angular head velocity and acceleration. We have to do this before we merge with the gaze
            # data (and thus interpolate head movement) to properly clean measurement errors due to wrapping from
            # 180/360 to 0 degree.
            temp_head_dict = copy.deepcopy(head_dict)
            temp_head_dict['headLatLon'] = np.mod((temp_head_dict['headLatLon'] + np.array([90.0, 180.0])) \
                                                  * np.array([-1.0, 1.0]), np.array([180.0, 360.0]))
            ang_head_velo, ang_head_acc = get_ang_va(temp_head_dict, 'headLatLon')
            head_dict.update({'ang_head_velo':ang_head_velo, 'ang_head_acc':ang_head_acc})

            # Join the gaze and head logs by timestamps
            joined_log = get_closest_timestamp_merge(head_dict, clean_gaze_dict, mode='max', suffixes=[])
            joined_log['headLatLon'] = np.mod((joined_log['headLatLon'] + np.array([90.0, 180.0])) *
                                              np.array([-1.0, 1.0]), np.array([180.0, 360.0]))

            # Get the latitude, longitude offset of the gaze relative to the head orientation
            # gaze_lat_lon_offset = get_gaze_lat_lon_offset(joined_log['both_coords'], joined_log['headTilt'])
            moving_average_filter = np.array(4*[1/4.0])

            gaze_lat_lon = get_gaze_lat_lon(joined_log['both_coords'], joined_log['headLatLon'], joined_log['headTilt'])
            gaze_lat_lon[:, 0] = np.convolve(gaze_lat_lon[:, 0], moving_average_filter, mode='same')
            gaze_lat_lon[:, 1] = np.convolve(gaze_lat_lon[:, 1], moving_average_filter, mode='same')

            gaze_lat_lon_offset = gaze_lat_lon - joined_log['headLatLon']
            gaze_lat_lon_offset[gaze_lat_lon_offset[:, 1]>=100.0, 1] -= 360.0
            gaze_lat_lon_offset[gaze_lat_lon_offset[:, 1]<=-100.0, 1] += 360.0

            # Interpolate wrong measurements
            wrong_measurements = np.abs(gaze_lat_lon_offset[:, 1])>100.0
            print(np.sum(wrong_measurements)/float(len(gaze_lat_lon_offset[:, 1])))
            gaze_lat_lon_offset[:, 1] = interpolate_nan_rows(gaze_lat_lon_offset[:, 1], wrong_measurements)

            assert(not np.any(np.abs(gaze_lat_lon_offset[:, 1])>100.0))

            joined_log.update({'gaze_lat_lon_offset':gaze_lat_lon_offset, 'gaze_lat_lon':gaze_lat_lon})

            if inspect:
                plt.scatter(joined_log['gaze_lat_lon'][:, 1], joined_log['gaze_lat_lon'][:, 0])
                plt.scatter(joined_log['headLatLon'][:, 1], joined_log['headLatLon'][:, 0], color='r')
                plt.show()

            # Get the eye fixations
            joined_log['gaze_fixations'], \
            joined_log['gaze_fixations_single'], \
            joined_log['gaze_fixations_ind'], \
            joined_log['gaze_fixations_raw'] = get_eye_fixations(joined_log, mode=fix_mode)

            # Get the angular head+gaze velocity and acceleration
            ang_gaze_velo, ang_gaze_acc  = get_ang_va(joined_log, column='gaze_lat_lon')
            joined_log['ang_gaze_velo'] = ang_gaze_velo
            joined_log['ang_gaze_acc'] = ang_gaze_acc

            # Get the angular gaze-only velocity and acceleration
            ang_gaze_velo, ang_gaze_acc = get_ang_va(joined_log, column='gaze_lat_lon_offset', filter_wrappers=False)
            joined_log['ang_gaze_offset_velo'] = ang_gaze_velo
            joined_log['ang_gaze_offset_acc'] = ang_gaze_acc

            starting_viewpoint = get_starting_viewpoint(joined_log)

            if inspect:
                data_analysis.visualize_combined_log([joined_log])

            summary_dict = {'data':joined_log, 'viewpoint_idx':starting_viewpoint, 'experiment_id':experiment_id}

            if save_mat:
                mat_path = os.path.join(out_dir, cubemap_dir, os.path.basename(gaze_logfile).rsplit('.', 1)[0])
                io.savemat(mat_path, summary_dict)
            all_runs.append(summary_dict)

            print('\n')

        with open(os.path.join(out_dir, '%s.pck'%cubemap_dir), 'w') as data_file:
            pickle.dump(pd.DataFrame(all_runs), data_file)

    print("Finished preprocessing. Discarded a total of %d out of %d runs"%(num_discarded, total_runs))

    return pd.DataFrame(all_runs)

def test_pipeline():
    def test_eye_join():
        timestamps = np.arange(0, 31.0, 0.004)
        timestamps[1::2] += np.random.uniform(0.001, 0.004, len(timestamps)/2)

        left_length = np.ceil(len(timestamps)/2.0).astype(int)
        right_length = np.floor(len(timestamps)/2.0).astype(int)

        left_timestamps = timestamps[0::2]
        right_timestamps = timestamps[1::2]

        left_eye_coords = np.column_stack( [np.sin(left_timestamps * 2 * np.pi), np.cos(left_timestamps * 2 * np.pi)]) + np.array([0, -0.05])
        right_eye_coords = np.column_stack( [np.sin(right_timestamps * 2 * np.pi), np.cos(right_timestamps * 2 * np.pi)]) + np.array([0, +0.05])
        both_eyes = np.zeros((len(timestamps), 2))
        both_eyes[0:len(left_eye_coords)*2:2, :] = left_eye_coords
        both_eyes[1:len(right_eye_coords)*2 + 1:2, :] = right_eye_coords

        left_eye_confs = np.random.uniform(0.85, 1., size=left_length)
        right_eye_confs = np.random.uniform(0.85, 1., size=right_length)
        both_confs = np.zeros(len(timestamps))
        both_confs[0:len(left_eye_confs)*2:2] = left_eye_confs
        both_confs[1:len(right_eye_confs)*2 + 1:2] = right_eye_confs

        eye_list = np.array(['left' if idx%2 == 0 else 'right' for idx in xrange(len(timestamps))])

        gaze_dict = {'timestamp':timestamps, 'confidence':both_confs, 'coords':both_eyes, 'eye':eye_list}

        full_gaze_dict = join_left_right_gaze(gaze_dict)
        assert(len(full_gaze_dict['both_coords']) == max(left_length, right_length))
        assert(np.mean(np.abs(full_gaze_dict['both_coords'] - np.mean(np.stack([np.roll(left_eye_coords, -2), right_eye_coords]), axis=0))) < 1/1000.0 )
        assert(np.mean(np.abs(full_gaze_dict['timestamp'] - right_timestamps)) < 1/1000.0)

        low_confs = full_gaze_dict['both_confidence']<0.9

        clean_gaze_dict = clean_gaze_records(full_gaze_dict, confidence_threshold=0.5)
        print("Average low-confidence interpolation error is %f"%
              np.median(np.abs(clean_gaze_dict['both_coords'][low_confs] - full_gaze_dict['both_coords'][low_confs])))

    def test_angular_velocity():
        timestamps = np.arange(0, 30.0, 0.008)
        test_coords = np.mod(np.column_stack( [np.sin(timestamps*0.5*np.pi)*400, np.cos(timestamps*0.5*np.pi+2)*500]), 360.0)

        test_dict = {'timestamp':timestamps, 'both_coords':test_coords}
        result = get_ang_va(test_dict, 'both_coords')[0]

        plt.plot(timestamps, test_coords[:, 0])
        # plt.plot(timestamps, test_coords[:, 1])
        plt.plot(timestamps, result[:, 0])
        # plt.plot(timestamps, result[:, 1])
        plt.show()


    def test_gaze_tilt():
        test_eye_coords = []

        for i in np.arange(0.0, 1.1, 0.25):
            for j in np.arange(0.0, 1.1, 0.25):
                test_eye_coords.append(np.array([i, j]))

        test_eye_coords = np.stack(test_eye_coords)
        head_tilts = np.random.uniform(-10, 10, len(test_eye_coords))

        tilted_coords = gaze_account_head_tilt(test_eye_coords, head_tilts)

        for label, x, y in zip(head_tilts, test_eye_coords[:, 0], test_eye_coords[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.scatter(test_eye_coords[:, 0], test_eye_coords[:, 1])
        plt.scatter(tilted_coords[:, 0], tilted_coords[:, 1])
        plt.show()


    def test_gaze_lat_lon():
        test_eye_coords = []

        for i in np.arange(0.0, 1.1, 0.25):
            for j in np.arange(0.0, 1.1, 0.25):
                test_eye_coords.append(np.array([i, j]))

        test_eye_coords = np.stack(test_eye_coords)
        center_lat_lons = np.ones((len(test_eye_coords), 2)) * np.array([90.0, 180.0])
        head_tilts = np.zeros(len(test_eye_coords))

        gaze_lat_lons = get_gaze_lat_lon(test_eye_coords, center_lat_lons, head_tilts)

        for label, x, y in zip(test_eye_coords, gaze_lat_lons[:, 1], gaze_lat_lons[:, 0]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.scatter(gaze_lat_lons[:, 1], gaze_lat_lons[:, 0])
        plt.show()


    test_eye_join()
    test_angular_velocity()
    test_gaze_tilt()
    test_gaze_lat_lon()

    return


if __name__=='__main__':
    # data = parse_head_log('/media/vsitzmann/New Volume/Programming/datasets/SIGGRAPH/Experiments/cubemap_0000/0727_0103_E0105_head_log.csv')
    # data = parse_gaze_log('/media/vsitzmann/New Volume/Programming/datasets/SIGGRAPH/Experiments/cubemap_0000/0727_0103_E0105_gaze_log.csv')
    debug = consolidate_log_to_dfs('/home/vincent/Downloads/split', '/home/vincent/Downloads/split/Computed', inspect=False, mean_conf_thresh=0.7)
    # test_pipeline()
