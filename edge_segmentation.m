function [edges, mask, object, info] = edge_segmentation(image, method, threshold, radius, sigma)
%EDGE_SEGMENTATION Deteksi tepi dan pilih satu komponen objek dari konturnya.
arguments
    image
    method (1,1) string = "Sobel"
    threshold (1,1) double {mustBeInRange(threshold, 0, 1)} = 0.15
    radius (1,1) double {mustBeInteger, mustBeNonnegative, mustBeFinite} = 4
    sigma (1,1) double {mustBePositive, mustBeFinite} = 1.2
end

gray = image;
if size(gray, 3) == 3
    gray = rgb2gray(gray);
end
gray = im2double(gray);
switch method
    case "Canny"
        if threshold == 0
            edges = edge(gray, 'Canny', [], sigma);
        else
            edges = edge(gray, 'Canny', [0.4 * threshold, threshold], sigma);
        end
    case {"Laplace", "LoG"}
        if strcmp(method, "Laplace")
            kernel = [0 1 0; 1 -4 1; 0 1 0];
        else
            r = ceil(3 * sigma);
            [x, y] = meshgrid(-r:r);
            d2 = x.^2 + y.^2;
            kernel = (d2 - 2 * sigma^2) .* exp(-d2 / (2 * sigma^2));
            % Kernel terpotong harus tetap memberi respons nol pada bidang konstan.
            kernel = kernel - mean(kernel, 'all');
        end
        response = imfilter(gray, kernel, 'replicate');
        edges = relativeThreshold(zeroCrossingStrength(response), threshold);
    case {"Sobel", "Prewitt", "Roberts"}
        switch method
            case "Sobel"
                kx = [-1 0 1; -2 0 2; -1 0 1];
                ky = kx.';
            case "Prewitt"
                kx = [-1 0 1; -1 0 1; -1 0 1];
                ky = kx.';
            case "Roberts"
                kx = [1 0; 0 -1];
                ky = [0 1; -1 0];
        end
        gx = imfilter(gray, kx, 'replicate');
        gy = imfilter(gray, ky, 'replicate');
        edges = bwmorph(relativeThreshold(hypot(gx, gy), threshold), 'thin', Inf);
    otherwise
        error('IF4073:UnknownEdgeOperator', 'Operator tepi tidak dikenal: %s.', method);
end

boundary = edges;
if radius > 0
    boundary = imclose(boundary, strel('disk', radius, 0));
end
interior = imfill(boundary, 4, 'holes') & ~boundary;
if radius > 0
    % Satukan interior yang terbelah oleh garis tekstur tipis di dalam objek.
    interior = imclose(interior, strel('disk', min(radius, 5), 0));
end
minArea = max(4, round(numel(edges) * 0.002));
candidate = bestComponent(interior, minArea, 4);
if any(candidate, 'all')
    % Tambahkan kembali pita tepi yang langsung membatasi interior terpilih.
    boundaryWidth = max(1, min(radius, 3));
    mask = candidate | (imdilate(candidate, strel('disk', boundaryWidth, 0)) & boundary);
    strategy = "kontur tertutup";
else
    % Bila kontur terbuka, hull dari kelompok tepi terbesar menjadi perkiraan objek.
    candidate = bestComponent(boundary, 2, 8);
    if any(candidate, 'all')
        mask = contourHull(candidate);
        strategy = "perkiraan hull";
    else
        mask = candidate;
        strategy = "tepi tidak cukup";
    end
end

object = image;
for channel = 1:size(image, 3)
    plane = image(:, :, channel);
    plane(~mask) = 0;
    object(:, :, channel) = plane;
end
info = struct('Strategy', strategy, 'AreaFraction', nnz(mask) / numel(mask));
end

function mask = contourHull(contour)
mask = contour;
[y, x] = find(contour);
points = [x, y];
if size(points, 1) < 3 || rank(points - points(1, :)) < 2
    return
end
vertices = convhull(x, y);
mask = mask | poly2mask(x(vertices), y(vertices), size(mask, 1), size(mask, 2));
end

function result = relativeThreshold(strength, threshold)
peak = max(strength, [], 'all');
result = strength > 1e-10 & strength >= threshold * peak;
end

function strength = zeroCrossingStrength(response)
% Pasangan piksel yang berseberangan mendeteksi crossing tepat di nilai nol.
[height, width] = size(response);
padded = padarray(response, [1 1], 'replicate');
strength = zeros(height, width);
center = response;
directions = [0 1; 1 0; 1 1; 1 -1];
for direction = directions.'
    dr = direction(1);
    dc = direction(2);
    forward = padded((2:height+1) + dr, (2:width+1) + dc);
    back = padded((2:height+1) - dr, (2:width+1) - dc);
    crossing = center .* forward < 0 & abs(center) <= abs(forward);
    strength(crossing) = max(strength(crossing), abs(center(crossing) - forward(crossing)));
    crossing = center .* back < 0 & abs(center) < abs(back);
    strength(crossing) = max(strength(crossing), abs(center(crossing) - back(crossing)));
    atZero = abs(center) <= 1e-10 & back .* forward < 0;
    strength(atZero) = max(strength(atZero), abs(back(atZero) - forward(atZero)));
end
end

function selected = bestComponent(binary, minArea, connectivity)
components = bwconncomp(binary, connectivity);
properties = regionprops(components, 'Area', 'Centroid', 'BoundingBox');
[height, width] = size(binary);
selected = false(height, width);
bestScore = 0;
for index = 1:components.NumObjects
    region = properties(index);
    if region.Area < minArea
        continue
    end
    center = (region.Centroid - ([width, height] + 1) / 2) ./ [width, height];
    box = region.BoundingBox;
    borders = [box(1) <= 0.5, box(2) <= 0.5, ...
        box(1) + box(3) >= width + 0.5, box(2) + box(4) >= height + 0.5];
    aspect = max(box(3:4)) / min(box(3:4));
    elongatedPenalty = sqrt(max(1, aspect / 3));
    score = region.Area / ((1 + 2 * norm(center) + 2 * nnz(borders)) * elongatedPenalty);
    if score > bestScore
        bestScore = score;
        bestIndex = index;
    end
end
if bestScore > 0
    selected(components.PixelIdxList{bestIndex}) = true;
end
end
