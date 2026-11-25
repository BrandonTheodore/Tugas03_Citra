function [segmented, labels, info] = split_merge_segmentation(image, splitSigma, mergeDelta, mergeSigma, minSide)
%SPLIT_MERGE_SEGMENTATION Pemisahan rekursif dan penggabungan tetangga-4 CIELAB.
arguments
    image
    splitSigma (1,1) double {mustBeNonnegative, mustBeFinite} = 10
    mergeDelta (1,1) double {mustBeNonnegative, mustBeFinite} = 12
    mergeSigma (1,1) double {mustBeNonnegative, mustBeFinite} = 12
    minSide (1,1) double {mustBeInteger, mustBePositive, mustBeFinite} = 8
end

rgb = im2double(image);
if size(rgb, 3) == 1
    rgb = repmat(rgb, 1, 1, 3);
end
lab = rgb2lab(rgb);
[height, width, ~] = size(lab);
leaves = zeros(height, width, 'uint32');
capacity = floor(height / min(height, minSide)) * floor(width / min(width, minSide));
area = zeros(capacity, 1);
colorSum = zeros(capacity, 3);
squaredSum = zeros(capacity, 1);
leafCount = 0;
split(1, height, 1, width);
area = area(1:leafCount);
colorSum = colorSum(1:leafCount, :);
squaredSum = squaredSum(1:leafCount);
parent = (1:leafCount).';

% Setiap pasangan berasal dari piksel yang berbagi sisi secara langsung.
left = leaves(:, 1:end-1);
right = leaves(:, 2:end);
upper = leaves(1:end-1, :);
lower = leaves(2:end, :);
pairs = double([left(:), right(:); upper(:), lower(:)]);
pairs = pairs(pairs(:, 1) ~= pairs(:, 2), :);
pairs = unique(sort(pairs, 2), 'rows');

while ~isempty(pairs)
    means = colorSum ./ area;
    a = pairs(:, 1);
    b = pairs(:, 2);
    distance = vecnorm(means(a, :) - means(b, :), 2, 2);
    [~, order] = sort(distance);
    changed = false;
    for index = order.'
        first = root(a(index));
        second = root(b(index));
        if first == second
            continue
        end
        distance = norm(colorSum(first, :) / area(first) - colorSum(second, :) / area(second));
        if distance > mergeDelta
            continue
        end
        count = area(first) + area(second);
        sumAB = colorSum(first, :) + colorSum(second, :);
        squaresAB = squaredSum(first) + squaredSum(second);
        % Uji deviasi gabungan mencegah rangkaian merge melebihi homogenitas.
        if dispersion(count, sumAB, squaresAB) > mergeSigma
            continue
        end
        if area(first) < area(second)
            [first, second] = deal(second, first);
        end
        parent(second) = first;
        area(first) = count;
        colorSum(first, :) = sumAB;
        squaredSum(first) = squaresAB;
        changed = true;
    end
    if ~changed
        break
    end
    for index = 1:leafCount
        parent(index) = root(index);
    end
    pairs = reshape(parent(pairs(:)), size(pairs));
    pairs = pairs(pairs(:, 1) ~= pairs(:, 2), :);
    pairs = unique(sort(pairs, 2), 'rows');
end

for index = 1:leafCount
    parent(index) = root(index);
end
[regions, ~, labelByLeaf] = unique(parent);
labels = reshape(uint32(labelByLeaf(leaves(:))), height, width);
finalCount = numel(regions);
% Warna keluaran adalah rerata RGB asli pada komponen hasil akhir.
flatRgb = reshape(rgb, [], 3);
meanRgb = zeros(finalCount, 3);
for channel = 1:3
    meanRgb(:, channel) = accumarray(double(labels(:)), flatRgb(:, channel), [finalCount, 1], @mean);
end
segmented = reshape(meanRgb(labels(:), :), height, width, 3);
info = struct('Leaves', leafCount, 'Regions', finalCount);

    function split(r1, r2, c1, c2)
        blockHeight = r2 - r1 + 1;
        blockWidth = c2 - c1 + 1;
        count = blockHeight * blockWidth;
        pixels = reshape(lab(r1:r2, c1:c2, :), [], 3);
        sumLab = sum(pixels, 1);
        squares = sum(pixels.^2, 'all');
        canSplitRows = blockHeight >= 2 * minSide;
        canSplitColumns = blockWidth >= 2 * minSide;
        if (~canSplitRows && ~canSplitColumns) || dispersion(count, sumLab, squares) <= splitSigma
            leafCount = leafCount + 1;
            leaves(r1:r2, c1:c2) = leafCount;
            area(leafCount) = count;
            colorSum(leafCount, :) = sumLab;
            squaredSum(leafCount) = squares;
            return
        end
        rowStops = [r1, r2 + 1];
        columnStops = [c1, c2 + 1];
        if canSplitRows
            rowStops = [r1, r1 + floor(blockHeight / 2), r2 + 1];
        end
        if canSplitColumns
            columnStops = [c1, c1 + floor(blockWidth / 2), c2 + 1];
        end
        for row = 1:numel(rowStops)-1
            for col = 1:numel(columnStops)-1
                split(rowStops(row), rowStops(row+1)-1, columnStops(col), columnStops(col+1)-1);
            end
        end
    end

    function representative = root(index)
        while parent(index) ~= index
            parent(index) = parent(parent(index));
            index = parent(index);
        end
        representative = index;
    end
end

function value = dispersion(count, colorSum, squaredSum)
variance = squaredSum / count - sum((colorSum / count).^2, 2);
% Toleransi kecil menangani galat pembulatan pada region warna datar.
value = sqrt(max(0, variance - 1e-10));
end
