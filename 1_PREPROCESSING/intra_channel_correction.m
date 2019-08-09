function out = intra_channel_correction(input_dir, output_dir, disk_size, brightfield)
tic

% check if output directory exis
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

image_fnames = dir(fullfile(input_dir, '*.tif'));
for i=1:size(image_fnames, 1)
    
    % read image and get the histogram of the original image
    t_in = Tiff(fullfile(input_dir, image_fnames(i).name), 'r+');
    im = read(t_in);
    close(t_in);
    
    % if this is not the brightfield channel
    if ~contains(image_fnames(i).name, int2str(brightfield))
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
    write_bigtiff(im, fullfile(output_dir, image_fnames(i).name));
    
end

fprintf('pipeline finished successfully in %.1f mins.\n', toc/60)
out = 1;
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

