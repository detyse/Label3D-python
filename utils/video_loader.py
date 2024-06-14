# maybe a class or a function to load multiple videos into finite frames
# 

import cv2
import numpy as np


# consider of just output a list of frames, we just using a function to do the job
def load_videos(video_file_list, label_num):
    '''
    the frame number not aligned situation is not considered
    it is should be fine with a single video (which is the normal situation)
    and assume all the videos want the same frame number
    '''
    frame_num_list = []
    frame_index_list = []
    frames = []
    for video_file in video_file_list:
        cap = cv2.VideoCapture(video_file)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num_list.append(frame_num)

        indexes = np.linspace(0, frame_num-1, label_num, dtype=int)
        frame_index_list.append(indexes)

        # if the frame number is index, then get the frame
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)

            ret = cap.grab()
            if ret:
                ret, frame = cap.retrieve()
                if ret:
                    frames.append(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(rgb_frame)
        cap.release()

    return frames


# random sample the video frames, save the frames and the index
# all the views
def random_sample(video_file_list, sample_num, repeat_num, save_path):
    # get the frame number
    cap = cv2.VideoCapture(video_file_list[0])
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # get random index
    indexes = np.random.choice(frame_num, sample_num, replace=False)
    
    # add one of the index to the list
    index = indexes[0]
    indexes.append(*[index for _ in range(repeat_num)])
    
    # shuffle the index
    np.random.shuffle(indexes)

    # get the frames
    frames = []
    for video_file in video_file_list:
        cap = cv2.VideoCapture(video_file)
        for index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)

            ret = cap.grab()
            if ret:
                ret, frame = cap.retrieve()
                if ret:
                    frames.append(frame)
        cap.release()
    return  



# this file could run individually to sample the videos

# if __name__ == "__main__":
#     main()