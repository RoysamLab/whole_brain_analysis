function errors = intra_channel_correction(input_dir, output_dir, varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% intra_channel_correction function corrects the images in 'input_dir' and
% save them in 'output_dir'. This function corrects for AUTOFLUORESCENCE,
% NON-UNIFORM ILLUMINATION, PHOTOBLEACHING, IMAGE ARTIFACTS.
% This function save a script for corrected channels in the parent folder of 'output_dir'
% Input arguments:
% input_dir: /path/to/input/directory
% output_dir: /path/to/output/directory
% disk_size: size of the morphological disks (default = [20, 40])
% script_file: /path/to/csv/script/for/channels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% If there is no error at the end of function, the variable changes to 0
% for output of the function
errors = 1;

% check if output directory exis
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

% read arguments
if nargin == 2
    script_table = create_script_table(input_dir);
    disk_size = [20, 40];
elseif  nargin == 3
    script_table = create_script_table(input_dir);
    disk_size = varargin{1};
elseif nargin == 4
    disk_size = varargin{1};
    script_table = readtable(varargin{2});
end

for i=1:size(script_table, 1)

    % read image and get the histogram of the original image
    t_in = Tiff(fullfile(input_dir, script_table.filename{i}), 'r+');
    im = read(t_in);
    close(t_in);

    if strcmpi(script_table.('intraChannelCorrection'){i}, 'yes')
        % calculate the background using morphological opening
        background = im;
        for j = 1:length(disk_size)
            % create disk for morphological opening and closing
            se = strel('disk',disk_size(j));
            % Alternative Sequential filter
            background = imopen(background,se);
        end

        % extract the background from original image
        im = im - background;

        % normalize to 0-65535
        im = double(im);
        im = uint16(im - min(im(:)))*(65535 / (max(im(:)) - min(im(:))));
    end

% write image to disk
write_bigtiff(im, fullfile(output_dir, script_table.filename{i}));

end

% save script in parent directory if unsupervised (no script input)
if nargin ~= 5
    [parentdir,~,~]=fileparts(output_dir);
    writetable(script_table, fullfile(parentdir,'script.csv'), ...
               'Delimiter',',');
end

% no error...
errors = 0;
end

function write_bigtiff(image, name)
%WRITE_BIGTIFF writes bigtiff images
t = Tiff(name, 'w');
setTag(t, 'Photometric', 1)
setTag(t, 'BitsPerSample', 16)
setTag(t, 'ImageLength', size(image, 1))
setTag(t, 'ImageWidth', size(image, 2))
setTag(t, 'PlanarConfiguration', 1)
setTag(t,'RowsPerStrip',1)
write(t,image);
close(t);
end

function script_table = create_script_table(input_dir)
% create script table like this
%    filename      intra_channel_correction
%   ___________    ________________________
%   'R1C0.tif'              'Yes'
%   'R1C1.tif'              'Yes'
%   'R1C10.tif'             'Yes'
image_fnames = dir(fullfile(input_dir, '*.tif'));
filename = {image_fnames(:).name}';
intra_channel_correction = cell(size(filename, 1), 1);
intra_channel_correction(:) = {'Yes'};
script_table = table(filename, intra_channel_correction);
end
