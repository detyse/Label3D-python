clear all

folder = 'H:\WEIYU\YW_2024-10-23';
%find out all of the dannce file include subfolders
file_list = dir([folder filesep '**/*_py_annotation.mat']);
%% 

for i = 1:length(file_list)
    sub_file_name = file_list(i).name;
    sub_file_folder = file_list(i).folder;
    load([sub_file_folder filesep sub_file_name])
    sync = sync';
    labelData = labelData';
    for k = 1:6
        labelData{k}.data_2d = labelData{k}.data_2d(:, [1:14 17:44]);
        labelData{k}.data_3d = labelData{k}.data_3d(:, [1:21 25:66]);
    end
    camnames = transform_camname(camnames);
    params = transform_params(params);
    save([sub_file_folder filesep sub_file_name(1:21) 'ridTail_dannce.mat'], "camnames","sync","params","labelData")
end


function transformed_camnames = transform_camname(camnames)
    s = size(camnames);
    if s(1) == 6 && s(2) == 7
    else
        error("camname size unmatch.")
    end
    
    transformed_camnames = cell(1,6);
    for i = 1:6
        transformed_camnames{i} = camnames(i,:);
    end
end


function transformed_params = transform_params(params)
    s = length(params);
    if s(1) == 6
    else
        error("params size unmatch.")
    end
    
    transformed_params = cell(6,1);
    for i = 1:6
        transformed_params{i} = params(i);
    end
end
