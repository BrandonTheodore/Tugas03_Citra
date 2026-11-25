function image = read_segmentation_image(path)
%READ_SEGMENTATION_IMAGE Baca citra RGB, abu-abu, atau palet dari berkas.
[image, map] = imread(path);
if ~isempty(map)
    image = im2uint8(ind2rgb(image, map));
end
if size(image, 3) ~= 1 && size(image, 3) ~= 3
    error('IF4073:UnsupportedImage', 'Citra harus memiliki satu atau tiga kanal.');
end
end
