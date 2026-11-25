function app = segmentation_app()
%SEGMENTATION_APP GUI MATLAB untuk tiga metode segmentasi IF4073.
root = fileparts(mfilename('fullpath'));
if exist('rgb2lab', 'file') == 0 || exist('edge', 'file') == 0
    error('IF4073:MissingToolbox', 'Instal Image Processing Toolbox untuk menjalankan aplikasi.');
end

app = uifigure('Name', 'Segmentasi Citra IF4073', 'Position', [70 70 1160 730], ...
    'CloseRequestFcn', @closeApp);
layout = uigridlayout(app, [3 1], 'RowHeight', {38, '1x', 25}, 'Padding', [14 10 14 10]);
toolbar = uigridlayout(layout, [1 2], 'ColumnWidth', {110, '1x'}, 'Padding', [0 0 0 0]);
openButton = uibutton(toolbar, 'Text', 'Buka citra', 'ButtonPushedFcn', @openImage);
fileLabel = uilabel(toolbar, 'Text', 'Pilih citra masukan.');
tabs = uitabgroup(layout);
status = uilabel(layout);
image = [];
job = [];
poller = timer('ExecutionMode', 'fixedSpacing', 'Period', 1, 'TimerFcn', @pollPython);

edgeTab = uitab(tabs, 'Title', 'Deteksi Tepi');
edgeLayout = uigridlayout(edgeTab, [2 1], 'RowHeight', {70, '1x'});
edgeControls = uigridlayout(edgeLayout, [2 5], 'RowHeight', {21, 31}, ...
    'ColumnWidth', {112, 130, 122, 138, 137}, 'Padding', [0 0 0 0]);
operators = {'Laplace', 'LoG', 'Sobel', 'Prewitt', 'Roberts', 'Canny'};
uilabel(edgeControls, 'Text', 'Operator');
edgeMethod = uidropdown(edgeControls, 'Items', operators, 'Value', 'Sobel');
place(edgeMethod, 2, 1);
edgeThreshold = numericControl(edgeControls, 2, 'Ambang tepi', 0.15, [0 1], false);
closeRadius = numericControl(edgeControls, 3, 'Radius closing (px)', 4, [0 40], true);
edgeSigma = numericControl(edgeControls, 4, 'Sigma LoG / Canny', 1.2, [0.1 10], false);
edgeButton = uibutton(edgeControls, 'Text', 'Segmentasi objek', 'ButtonPushedFcn', @runEdge);
place(edgeButton, 2, 5);
edgeAxes = imageAxes(edgeLayout, ["Citra asli", "Tepi", "Objek"]);

splitTab = uitab(tabs, 'Title', 'Split and Merge');
splitLayout = uigridlayout(splitTab, [2 1], 'RowHeight', {70, '1x'});
splitControls = uigridlayout(splitLayout, [2 5], 'RowHeight', {21, 31}, ...
    'ColumnWidth', {112, 120, 120, 145, 136}, 'Padding', [0 0 0 0]);
splitSigma = numericControl(splitControls, 1, 'Deviasi split', 10, [0 200], false);
mergeDelta = numericControl(splitControls, 2, 'Jarak warna merge', 12, [0 300], false);
mergeSigma = numericControl(splitControls, 3, 'Deviasi merge', 12, [0 200], false);
minSide = numericControl(splitControls, 4, 'Ukuran minimum (px)', 8, [1 1024], true);
splitButton = uibutton(splitControls, 'Text', 'Segmentasi region', 'ButtonPushedFcn', @runSplit);
place(splitButton, 2, 5);
splitAxes = imageAxes(splitLayout, ["Citra asli", "Split and Merge"]);

semanticTab = uitab(tabs, 'Title', 'Semantik CamVid');
semanticLayout = uigridlayout(semanticTab, [5 1], 'RowHeight', {92, 32, '1x', 56, 88});
paths = uigridlayout(semanticLayout, [3 3], 'RowHeight', {26, 26, 26}, ...
    'ColumnWidth', {102, '1x', 102}, 'Padding', [0 0 0 0]);
defaultPython = 'python';
if ispc
    venvPython = fullfile(root, '.venv', 'Scripts', 'python.exe');
else
    venvPython = fullfile(root, '.venv', 'bin', 'python');
end
if isfile(venvPython)
    defaultPython = venvPython;
end
place(uilabel(paths, 'Text', 'Python'), 1, 1);
pythonPath = uieditfield(paths, 'text', 'Value', defaultPython);
place(pythonPath, 1, 2);
place(uibutton(paths, 'Text', 'Pilih Python', 'ButtonPushedFcn', @choosePython), 1, 3);
place(uilabel(paths, 'Text', 'Checkpoint'), 2, 1);
checkpointPath = uieditfield(paths, 'text', 'Value', fullfile(root, 'output', 'camvid', 'camvid.pth'));
place(checkpointPath, 2, 2);
place(uibutton(paths, 'Text', 'Pilih model', 'ButtonPushedFcn', @chooseCheckpoint), 2, 3);
place(uilabel(paths, 'Text', 'Folder CamVid'), 3, 1);
dataPath = uieditfield(paths, 'text', 'Value', fullfile(root, 'data', 'CamVid'));
place(dataPath, 3, 2);
place(uibutton(paths, 'Text', 'Pilih folder', 'ButtonPushedFcn', @chooseData), 3, 3);

actions = uigridlayout(semanticLayout, [1 3], 'ColumnWidth', {130, 108, 100}, 'Padding', [0 0 0 0]);
predictButton = uibutton(actions, 'Text', 'Prediksi citra', 'ButtonPushedFcn', @(~, ~) runPython("predict"));
trainButton = uibutton(actions, 'Text', 'Latih model', 'ButtonPushedFcn', @(~, ~) runPython("train"));
stopButton = uibutton(actions, 'Text', 'Hentikan', 'Enable', 'off', 'ButtonPushedFcn', @stopPython);
semanticAxes = imageAxes(semanticLayout, ["Citra asli", "Prediksi kelas", "Overlay"]);
legendLayout = uigridlayout(semanticLayout, [2 6], 'Padding', [0 0 0 0]);
legendLabels = gobjects(1, 11);
for index = 1:11
    legendLabels(index) = uilabel(legendLayout, 'HorizontalAlignment', 'center', 'Visible', 'off');
end
processLog = uitextarea(semanticLayout, 'Editable', 'off', 'FontName', 'Consolas', ...
    'Value', {'Log pelatihan dan inferensi Python.'});
openImageButtons(false);

    function openImage(~, ~)
        [name, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Berkas citra'}, 'Buka citra');
        if isequal(name, 0)
            return
        end
        try
            image = read_segmentation_image(fullfile(folder, name));
            fileLabel.Text = sprintf('%s  |  %d × %d px', name, size(image, 2), size(image, 1));
            resetView(edgeAxes, ["Citra asli", "Tepi", "Objek"]);
            resetView(splitAxes, ["Citra asli", "Split and Merge"]);
            resetView(semanticAxes, ["Citra asli", "Prediksi kelas", "Overlay"]);
            set(legendLabels, 'Visible', 'off');
            status.Text = '';
            openImageButtons(true);
        catch exception
            showError(exception, 'Gagal membuka citra');
        end
    end

    function resetView(axes, titles)
        showImage(axes(1), image, titles(1));
        for axisIndex = 2:numel(axes)
            cla(axes(axisIndex));
            title(axes(axisIndex), titles(axisIndex));
        end
    end

    function openImageButtons(enabled)
        set([edgeButton, splitButton], 'Enable', enabled);
        predictButton.Enable = enabled && isempty(job);
    end

    function runEdge(~, ~)
        try
            [edges, ~, object, info] = edge_segmentation(image, edgeMethod.Value, ...
                edgeThreshold.Value, closeRadius.Value, edgeSigma.Value);
            showImage(edgeAxes(2), edges, "Tepi " + edgeMethod.Value);
            showImage(edgeAxes(3), object, 'Objek');
            status.Text = sprintf('Mask objek %.1f%%  |  %s', 100 * info.AreaFraction, info.Strategy);
        catch exception
            showError(exception, 'Deteksi tepi gagal');
        end
    end

    function runSplit(~, ~)
        try
            [segmented, ~, info] = split_merge_segmentation(image, splitSigma.Value, ...
                mergeDelta.Value, mergeSigma.Value, minSide.Value);
            showImage(splitAxes(2), segmented, 'Split and Merge');
            status.Text = sprintf('%d region setelah split  |  %d region setelah merge', info.Leaves, info.Regions);
        catch exception
            showError(exception, 'Split and Merge gagal');
        end
    end

    function choosePython(~, ~)
        [name, folder] = uigetfile('*', 'Pilih executable Python');
        if ~isequal(name, 0)
            pythonPath.Value = fullfile(folder, name);
        end
    end

    function chooseCheckpoint(~, ~)
        [name, folder] = uigetfile('*.pth', 'Pilih checkpoint CamVid');
        if ~isequal(name, 0)
            checkpointPath.Value = fullfile(folder, name);
        end
    end

    function chooseData(~, ~)
        folder = uigetdir(root, 'Pilih folder induk CamVid');
        if ~isequal(folder, 0)
            dataPath.Value = folder;
        end
    end

    function runPython(action)
        if ~isempty(job)
            return
        end
        checkpoint = strtrim(checkpointPath.Value);
        if isempty(checkpoint) || (action ~= "train" && ~isfile(checkpoint))
            uialert(app, 'Pilih checkpoint CamVid (.pth) atau latih model terlebih dahulu.', 'Checkpoint CamVid');
            return
        end
        if action == "train" && ~isfolder(dataPath.Value)
            uialert(app, 'Pilih folder induk yang berisi citra dan anotasi CamVid.', 'Folder CamVid');
            return
        end
        if action == "train" && isfile(checkpoint)
            confirmation = uiconfirm(app, 'Checkpoint akan diganti jika model baru berhasil dilatih.', ...
                'Latih model CamVid', 'Options', {'Latih dan ganti', 'Batal'}, 'DefaultOption', 2, 'CancelOption', 2);
            if ~strcmp(confirmation, 'Latih dan ganti')
                return
            end
        end

        workFolder = tempname;
        mkdir(workFolder);
        logPath = fullfile(workFolder, 'process.log');
        maskPath = fullfile(workFolder, 'mask.png');
        metadataPath = fullfile(workFolder, 'classes.json');
        argumentsToRun = [action, "--checkpoint", string(checkpoint)];
        switch action
            case "predict"
                inputPath = fullfile(workFolder, 'image.png');
                argumentsToRun = [argumentsToRun, "--image", inputPath, "--output", maskPath, "--metadata", metadataPath];
            case "train"
                argumentsToRun = [argumentsToRun, "--data", string(dataPath.Value)];
        end
        try
            if action == "predict"
                imwrite(im2uint8(image), inputPath);
            end
            launchProcess(strtrim(pythonPath.Value), fullfile(root, 'camvid_job.py'), argumentsToRun, workFolder);
            job = struct('Action', action, 'Folder', workFolder, 'Status', fullfile(workFolder, 'status.txt'), ...
                'Log', logPath, 'Mask', maskPath, 'Metadata', metadataPath, 'Stopped', false, 'CloseWhenDone', false);
            start(poller);
            set([predictButton, trainButton], 'Enable', 'off');
            if action == "predict"
                openButton.Enable = 'off';
                resetView(semanticAxes, ["Citra asli", "Prediksi kelas", "Overlay"]);
                set(legendLabels, 'Visible', 'off');
            end
            stopButton.Enable = 'on';
            processLog.Value = {'Menjalankan Python…'};
        catch exception
            if isempty(job)
                rmdir(workFolder, 's');
            else
                stopPython([], []);
            end
            showError(exception, 'Python gagal dijalankan');
        end
    end

    function pollPython(~, ~)
        try
            if isfile(job.Log)
                lines = splitlines(strtrim(string(fileread(job.Log))));
                if strlength(lines(end)) > 0
                    processLog.Value = cellstr(lines(max(1, end-9):end));
                end
            end
            if ~isfile(job.Status)
                return
            end
            if job.Stopped
                status.Text = 'Proses Python dihentikan.';
            elseif str2double(fileread(job.Status)) ~= 0
                uialert(app, strjoin(processLog.Value, newline), 'Proses Python gagal');
            elseif job.Action == "predict"
                displayPrediction();
                status.Text = 'Prediksi CamVid selesai.';
                processLog.Value = {'Prediksi dari checkpoint CamVid berhasil.'};
            else
                status.Text = 'Proses CamVid selesai. Lokasi hasil tersedia di log.';
            end
        catch exception
            if ~isfile(job.Status)
                return
            end
            showError(exception, 'Hasil CamVid gagal dibuka');
        end
        closeWhenDone = job.CloseWhenDone;
        finishJob();
        if closeWhenDone
            delete(poller);
            delete(app);
        end
    end

    function displayPrediction()
        mask = imread(job.Mask);
        metadata = jsondecode(fileread(job.Metadata));
        overlay = 0.52 * im2double(image) + 0.48 * im2double(mask);
        showImage(semanticAxes(2), mask, 'Prediksi kelas');
        showImage(semanticAxes(3), overlay, 'Overlay');
        for classIndex = 1:numel(metadata.classes)
            color = metadata.colors(classIndex, :) / 255;
            swatch = legendLabels(classIndex);
            swatch.Text = metadata.classes{classIndex};
            swatch.BackgroundColor = color;
            swatch.FontColor = repmat(double(dot(color, [0.299 0.587 0.114]) < 0.52), 1, 3);
            swatch.Visible = 'on';
        end
    end

    function stopPython(~, ~)
        if ~isempty(job) && ~isfile(job.Status)
            signal = fopen(fullfile(job.Folder, 'stop'), 'w');
            if signal < 0
                uialert(app, 'Berkas penghentian proses tidak dapat dibuat.', 'Gagal menghentikan Python');
                return
            end
            fclose(signal);
            job.Stopped = true;
            stopButton.Enable = 'off';
        end
    end

    function finishJob()
        stop(poller);
        [~, ~] = rmdir(job.Folder, 's');
        job = [];
        openButton.Enable = 'on';
        trainButton.Enable = 'on';
        stopButton.Enable = 'off';
        openImageButtons(~isempty(image));
    end

    function closeApp(~, ~)
        if ~isempty(job)
            if ~isfile(job.Status)
                if job.CloseWhenDone
                    return
                end
                confirmation = uiconfirm(app, 'Hentikan proses Python dan tutup aplikasi?', ...
                    'Proses sedang berjalan', 'Options', {'Hentikan dan tutup', 'Batal'}, ...
                    'DefaultOption', 2, 'CancelOption', 2);
                if ~strcmp(confirmation, 'Hentikan dan tutup')
                    return
                end
                if ~isempty(job) && ~isfile(job.Status)
                    stopPython([], []);
                    job.CloseWhenDone = job.Stopped;
                    return
                end
            end
            if ~isempty(job)
                finishJob();
            end
        end
        delete(poller);
        delete(app);
    end

    function showError(exception, heading)
        uialert(app, exception.message, heading);
    end
end

function field = numericControl(layout, column, caption, value, limits, integer)
place(uilabel(layout, 'Text', caption), 1, column);
field = uieditfield(layout, 'numeric', 'Value', value, 'Limits', limits);
field.RoundFractionalValues = integer;
place(field, 2, column);
end

function place(component, row, column)
component.Layout.Row = row;
component.Layout.Column = column;
end

function axes = imageAxes(layout, names)
grid = uigridlayout(layout, [1 numel(names)], 'Padding', [0 0 0 0]);
axes = gobjects(1, numel(names));
for index = 1:numel(names)
    ax = uiaxes(grid, 'XTick', [], 'YTick', [], 'Box', 'off');
    ax.Toolbar.Visible = 'off';
    disableDefaultInteractivity(ax);
    title(ax, names(index));
    axes(index) = ax;
end
end

function showImage(ax, data, caption)
imshow(data, 'Parent', ax);
title(ax, caption, 'Interpreter', 'none');
end

function launchProcess(executable, launcher, args, folder)
% Path dikirim melalui variabel lingkungan dan argumen proses melalui JSON.
file = fopen(fullfile(folder, 'arguments.json'), 'w', 'n', 'UTF-8');
if file < 0
    error('IF4073:PythonArguments', 'Berkas argumen Python tidak dapat dibuat.');
end
guard = onCleanup(@() fclose(file));
fprintf(file, '%s', jsonencode(args));
clear guard
if ispc
    command = ['powershell.exe -NoProfile -NonInteractive -Command ' ...
        '"$ErrorActionPreference = ''Stop''; & $env:IF4073_PYTHON $env:IF4073_LAUNCHER start $env:IF4073_JOB; exit $LASTEXITCODE"'];
else
    command = '"$IF4073_PYTHON" "$IF4073_LAUNCHER" start "$IF4073_JOB" </dev/null';
end
[statusCode, message] = system(command, 'IF4073_PYTHON', executable, ...
    'IF4073_LAUNCHER', launcher, 'IF4073_JOB', folder);
if statusCode ~= 0
    error('IF4073:PythonStart', '%s', strtrim(message));
end
end
