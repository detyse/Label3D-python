function labelData = synAnno4DANNCE(predPath, paramsPath, idx, annoSetPath)
% JZ @ 2024.9.4
% Convert annotation from Yulong Wang`s software (.npy) to the format accepted by DANNCE
% Syntax:
% synAnno4DANNCE("D:\Working Platform\Data\Changliang Liu Lab\Ethology\Annotation\T02_WC\Tongxuan Wei\joints3d.npy","D:\Working Platform\Data\Changliang Liu Lab\Ethology\Annotation\T02_WC\Tongxuan Wei\dannce_format_camera_params.mat")
% synAnno4DANNCE("D:\Working Platform\Data\Changliang Liu Lab\Ethology\T02_WC\Tongxuan Wei\Annotation\joints3d.npy","D:\Working Platform\Data\Changliang Liu Lab\Ethology\T02_WC\Tongxuan Wei\Annotation\dannce_format_camera_params.mat", [], "D:\Working Platform\Data\Changliang Liu Lab\Ethology\T02_WC\Tongxuan Wei\Annotation\annotationSet_2024-07-23_cocaine20__240807.mat")
% See nargins for optional parameters

%% TBD: fix expID identification

pred = permute(readNPY(predPath), [1 3 2]);
automated = false;
if nargin == 4  % Provide the path to "annotationSet_xxx.mat", and the output would be de-multiplexed and saved automatically
    automated = true;
end
if nargin < 3  % Either valid indices or subsampled videos, up to you. Ignored if annoSetPath provided.
    idx = 1:size(pred, 1);
end
params = load(paramsPath, 'params');
params = params.params;
labelData = cell(max(size(params), 1));

QCID = ~(sum(isnan(pred), [2, 3]) / size(pred, 2) > 3);  % Left <= 3 points
predP = reprojection(pred, params, [1440, 1080]);
predP(~QCID, :, :, :) = nan;
pred = reshape(pred, size(pred, 1), []);
pred(~QCID, :, :) = nan;

if automated
    t = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z');  % timestamp, sorry for the Z
    metaAnno = load(annoSetPath, 'exp', 'iteExp', 'iteSample', 'idx', 'copies');
    idx = metaAnno.idx;  % Indices across all experiments. But data are from single one.
    nSample = size(metaAnno.iteSample, 1);
    szExp = (metaAnno.copies+1) * nSample;
    if mod(size(idx, 1), szExp) ~= 0
        error("synAnno4DANNCE: Invalid annotation set file.")
    else
        %[~, expID] = max(conv2(zscore(double(iteExp), 0, 'all'), zscore(double(exp)), 'same'), [], 'all')
        %[expID, ~] = ind2sub(size(iteExp), expID)
        expID = 1;
        idx = idx(szExp*(expID-1)+1:szExp*expID);

        [savepath, annoSetName, ~] = fileparts(annoSetPath);
        annoSetName = replace(annoSetName, "annotationSet", "A");
        annoSetName = replace(annoSetName, "__", "_");
        for sample = 1:nSample
            idxSample = (metaAnno.copies+1)*(sample-1)+1:(metaAnno.copies+1)*sample;
            for view = 1:size(labelData, 1)
                labelData{view, 1}.data_2d = reshape(predP(idxSample,:,:,view), size(idxSample, 2), []); % t * 42
                labelData{view, 1}.data_3d = pred(idxSample, :, :);       % t * 63
                labelData{view, 1}.data_frame = idx(idxSample);       % 1 * t
                labelData{view, 1}.data_sampleID = (idx(idxSample)'-1)*40+1;   % t * 1, time
            end
            save(fullfile(savepath, annoSetName+"_"+replace(metaAnno.iteSample(sample, 1:end-1), " ", "")), "labelData", "params", "t")
        end
    end
else
    for view = 1:size(labelData, 1)
        labelData{view, 1}.data_2d = reshape(predP(:,:,:,view), size(pred, 1), []); % t * 42
        labelData{view, 1}.data_3d = pred;       % t * 63
        labelData{view, 1}.data_frame = idx;       % 1 * t
        labelData{view, 1}.data_sampleID = idx';   % t * 1, time
    end
end
end

function predP = reprojection(pred, params, rsl)
% JZ updated @ 2024.9.4
% Reproject the world coordinates for the labeled joints to
% pred: M x 3
if nargin == 2
    rsl = [1440, 1080];
end
predP = zeros(size(pred, 1), size(pred, 2)-1, size(pred, 3), max(size(params)));
for cameraID = 1:max(size(params))
    % tform = rigidtform3d(params{cameraID}.r, params{cameraID}.t);
    intrinsics = cameraIntrinsics([params{cameraID}.K(1, 1),params{cameraID}.K(2, 2)], params{cameraID}.K(3, 1:2), rsl, 'RadialDistortion', params{cameraID}.RDistort, 'TangentialDistortion', params{cameraID}.TDistort, 'Skew', params{cameraID}.K(2,1));
    predP(:,:,:,cameraID) = reshape(cell2mat(arrayfun(@(x) worldToImage(intrinsics,params{cameraID}.r,params{cameraID}.t',pred(:, :, x)), [1 : size(pred, 3)], 'UniformOutput', false)), size(pred, 1), 2, []);
end
end

function data = readNPY(filename)
% Function to read NPY files into matlab.
% *** Only reads a subset of all possible NPY files, specifically N-D arrays of certain data types.
% See https://github.com/kwikteam/npy-matlab/blob/master/tests/npy.ipynb for
% more.
[shape, dataType, fortranOrder, littleEndian, totalHeaderLength, ~] = readNPYheader(filename);
if littleEndian
    fid = fopen(filename, 'r', 'l');
else
    fid = fopen(filename, 'r', 'b');
end
try
    [~] = fread(fid, totalHeaderLength, 'uint8');
    % read the data
    data = fread(fid, prod(shape), [dataType '=>' dataType]);
    if length(shape)>1 && ~fortranOrder
        data = reshape(data, shape(end:-1:1));
        data = permute(data, [length(shape):-1:1]);
    elseif length(shape)>1
        data = reshape(data, shape);
    end
    fclose(fid);
catch me
    fclose(fid);
    rethrow(me);
end
end