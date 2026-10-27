'''
(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2016  Pupil Labs

 Distributed under the terms of the GNU Lesser General Public License (LGPL v3.0).
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

import numpy as np
import logging
# logging
logger = logging.getLogger(__name__)

class Fixation_Detector_Dispersion_Duration():
    '''
    This plugin classifies fixations and saccades by measuring dispersion and duration of gaze points


    Fixations general knowledge from literature review
        + Goldberg et al. - fixations rarely < 100ms and range between 200ms and 400ms in duration (Irwin, 1992 - fixations dependent on task between 150ms - 600ms)
        + Very short fixations are considered not meaningful for studying behavior - eye+brain require time for info to be registered (see Munn et al. APGV, 2008)
        + Fixations are rarely longer than 800ms in duration
            + Smooth Pursuit is exception and different motif
            + If we do not set a maximum duration, we will also detect smooth pursuit (which is acceptable since we compensate for VOR)
    Terms
        + dispersion (spatial) = how much spatial movement is allowed within one fixation (in visual angular degrees or pixels)
        + duration (temporal) = what is the minimum time required for gaze data to be within dispersion threshold?

    '''
    def __init__(self,max_dispersion = 1.0,min_duration = 0.15):
        self.min_duration = min_duration
        self.max_dispersion = max_dispersion
        # self.h_fov = h_fov
        # self.v_fov = v_fov

    def classify(self, gaze_data):
        '''
        classify fixations
        gaze_data: python list of dictionaries containing 'norm_pos', 'confidence', 'index' and 'timestamp' for each sample point
        '''
        sample_threshold = self.min_duration * 3 *.3 #lets assume we need data for at least 30% of the duration
        dispersion_threshold = self.max_dispersion
        duration_threshold = self.min_duration

        def dist_deg(p1,p2):
            return np.sqrt(((p1[0]-p2[0])*360.0)**2+((p1[1]-p2[1])*180.0)**2)

        fixations = []
        try:
            fixation_support = [gaze_data.pop(0)]
        except IndexError:
            logger.warning("This recording has no gaze data. Aborting")
            return
        while True:
            fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
            dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])

            if dispersion < dispersion_threshold and gaze_data:
                #so far all samples inside the threshold, lets add a new candidate
                fixation_support += [gaze_data.pop(0)]
            else:
                if gaze_data:
                    #last added point will break dispersion threshold for current candidate fixation. So we conclude sampling for this fixation.
                    last_sample = fixation_support.pop(-1)
                if fixation_support:
                    duration = fixation_support[-1]['timestamp'] - fixation_support[0]['timestamp']
                    if duration > duration_threshold and len(fixation_support) > sample_threshold:
                        #long enough for fixation: we classifiy this fixation candidate as fixation
                        #calculate character of fixation
                        fixation_centroid = sum([p['norm_pos'][0] for p in fixation_support])/len(fixation_support),sum([p['norm_pos'][1] for p in fixation_support])/len(fixation_support)
                        dispersion = max([dist_deg(fixation_centroid,p['norm_pos']) for p in fixation_support])
                        confidence = sum(g['confidence'] for g in fixation_support)/len(fixation_support)

                        # avg pupil size  = mean of (mean of pupil size per gaze ) for all gaze points of support
                        # avg_pupil_size =  sum([sum([p['diameter'] for p in g['base_data']])/len(g['base_data']) for g in fixation_support])/len(fixation_support)
                        new_fixation = {'id': len(fixations),
                                        'norm_pos':fixation_centroid,
                                        'base_data':fixation_support,
                                        'duration':duration,
                                        'dispersion':dispersion,
                                        'start_time':fixation_support[0]['timestamp'],
                                        'end_time':fixation_support[-1]['timestamp'],
                                        'start_frame_index':fixation_support[0]['index'],
                                        'mid_frame_index':fixation_support[len(fixation_support)/2]['index'],
                                        'end_frame_index':fixation_support[-1]['index'],
                                        # 'pix_dispersion':dispersion*self.pix_per_degree,
                                        # 'timestamp':fixation_support[0]['timestamp'],
                                        # 'pupil_diameter':avg_pupil_size,
                                        'confidence':confidence}

                        fixations.append(new_fixation)
                if gaze_data:
                    #start a new fixation candite
                    fixation_support = [last_sample]
                else:
                    break

        #gather some statisics for debugging and feedback.
        total_fixation_time  = sum([f['duration'] for f in fixations])
        # total_video_time = self.g_pool.timestamps[-1]- self.g_pool.timestamps[0]
        fixation_count = len(fixations)
        print("detected %s Fixations. Total duration of fixations: %0.2fsec total time of video %%0.2fsec "%(fixation_count,total_fixation_time))#,total_video_time))

        return fixations

        # now lets bin fixations into frames. Fixations may be repeated this way as they span muliple frames
        # fixations_by_frame = [[] for x in self.g_pool.timestamps]
        # for f in fixations:
        #     for idx in range(f['start_frame_index'],f['end_frame_index']+1):
        #         fixations_by_frame[idx].append(f)
        #
        # self.g_pool.fixations_by_frame = fixations_by_frame

    # def close(self):
    #     self.alive = False


    # def get_init_dict(self):
    #     return {'max_dispersion': self.max_dispersion, 'min_duration':self.min_duration, 'h_fov':self.h_fov, 'v_fov': self.v_fov,'show_fixations':self.show_fixations}
    #
    # def cleanup(self):
    #     self.deinit_gui()

