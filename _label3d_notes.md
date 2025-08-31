# Label3D notes

+ put the cam params and the skeleton into the output folder (and rename), for the dannce app to load

+ to distinguish the same data from different experiments, add the prefix for the data
(could be the folder name)

+ cam params: exp_name-camparams.mat

+ skeleton: skeleton_name-skeleton.json

+ TODO: 进入 label 后在关闭窗口时, 回到初始的文件选择窗口

## viewer update notes

+ not using the generated skeleton. but using skeleton and joint selected combine.

+ no more consider the mat file video generation.

+ if the video number not align with the number of cameras, assume the first n cameras are used.
