function imageProcessingApp()
    % 创建主图形界面
    fig = uifigure('Name', 'Image Processing App', 'Position', [100, 100, 800, 600]);

    % 创建一个网格布局，第一列为按钮区，第二列为图像区
    grid = uigridlayout(fig, [1, 2]);  % 1行2列布局
    grid.ColumnWidth = {'0.2x', '0.8x'};  % 第一列宽度为20%，第二列宽度为80%

    % 创建一个子布局来放按钮（第一列）
    buttonLayout = uigridlayout(grid, [9, 1]);  % 9个按钮行
    buttonLayout.RowHeight = repmat({'1x'}, 1, 9);  % 按钮行均分高度
    buttonLayout.Padding = [10 10 10 10];  % 设置padding，避免按钮和边缘紧贴

    % 创建一个子布局来放图像（第二列）
    axLayout = uigridlayout(grid, [1, 2]);  % 创建一个子布局，2列分别显示原图和处理后的图像
    axLayout.ColumnWidth = {'1x', '1x'};  % 两列宽度均分

    % 创建第一个子轴来显示原图（第一列）
    ax1 = axes(axLayout);  % 第一列，原图
    title(ax1, 'Original Image');

    % 创建第二个子轴来显示处理后的图像（第二列）
    ax2 = axes(axLayout);  % 第二列，处理后的图像
    title(ax2, 'Processed Image');

    % 图像变量（初始化为空）
    global img;
    img = [];
    global noisyImg;


     % 创建一个文本标签，用于显示噪声类型和参数（第二行）
    noise_info = uilabel(axLayout);
    noise_info.Layout.Row = 2;
    noise_info.Layout.Column = [1, 2];  % 跨两列显示
    noise_info.Text = '';  % 初始时无噪声信息
    noise_info.FontSize = 12;
    noise_info.HorizontalAlignment = 'center';
    axLayout.RowHeight = {'1x', '0.05x'};  % 第二行占用的高度较小

    % 初始化按钮状态（存储每个按钮的当前功能状态）
     global buttonState;
    buttonState = struct('equalize', false, 'contrastEnhance', false, 'zoomImage', false, ...
                         'rotateImage', false, 'addNoise', false, 'applyFilters',false, ...
                         'edgeDetection', false, 'featureExtraction', false, 'histogramMatch', false);


    % 加载图像按钮
    uibutton(buttonLayout, 'Text', '加载图像', 'ButtonPushedFcn', @(btn, event) loadImage(ax1, ax2));
    
    % 显示灰度直方图按钮
    uibutton(buttonLayout, 'Text', '显示直方图', 'ButtonPushedFcn', @(btn, event) showHistogram(ax1));
    
    % 直方图均衡化按钮
    uibutton(buttonLayout, 'Text', '直方图均衡化', 'ButtonPushedFcn', @(btn, event) toggleEqualize(ax1, ax2));

     % 直方图匹配按钮
    uibutton(buttonLayout, 'Text', '直方图匹配', 'ButtonPushedFcn', @(btn, event) toggleHistogramMatch(ax1, ax2));

    % 图像对比度增强按钮
    uibutton(buttonLayout, 'Text', '对比度增强', 'ButtonPushedFcn', @(btn, event) toggleContrastEnhance(ax1, ax2));

    % 图像缩放按钮
    uibutton(buttonLayout, 'Text', '图像缩放', 'ButtonPushedFcn', @(btn, event) toggleZoomImage(ax1, ax2));

    % 图像旋转按钮
    uibutton(buttonLayout, 'Text', '图像旋转', 'ButtonPushedFcn', @(btn, event) toggleRotateImage(ax1, ax2));
    
    % 添加噪声按钮
    uibutton(buttonLayout, 'Text', '添加噪声', 'ButtonPushedFcn', @(btn, event) toggleAddNoise(ax1, ax2, noise_info));

    % 添加滤波按钮
    uibutton(buttonLayout, 'Text', '滤波处理', 'ButtonPushedFcn', @(btn, event) applyFilter(ax1, ax2));

    % 边缘检测按钮
    uibutton(buttonLayout, 'Text', '边缘检测', 'ButtonPushedFcn', @(btn, event) toggleEdgeDetection(ax1, ax2));

    % 特征提取按钮
    uibutton(buttonLayout, 'Text', '特征提取', 'ButtonPushedFcn', @(btn, event) toggleFeatureExtraction(ax1, ax2));
end

% 加载图像函数
function loadImage(ax1, ax2)
    global img;
    [file, path] = uigetfile({'*.jpg;*.png;*.bmp','Image Files'}, 'Select an Image');
    if isequal(file, 0)
        return;
    end
    img = imread(fullfile(path, file));
    imshow(img, 'Parent', ax1);  % 显示原图
    imshow([], 'Parent', ax2);  % 清空处理后的图像
end

% 显示灰度直方图函数
function showHistogram(ax1)
    global img;

    if isempty(img)
        uialert(ax1, '请先加载一张图像！', '错误');
        return;
    end

    % 如果图像是彩色图像，则转换为灰度图像
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end

    % 计算灰度直方图
    [counts, binLocations] = imhist(img_gray);
    
    % 清空第二个子图，准备显示直方图
    axes(ax1);
    cla;
    
    % 绘制灰度直方图
    bar(binLocations, counts, 'BarWidth', 1, 'FaceColor', [0.7 0.7 0.7]);
    title('Gray Histogram');
    xlabel('Gray Level');
    ylabel('Frequency');
    xlim([0 255]);
end


% 直方图均衡化功能
function toggleEqualize(~, ax2)
    global img;
    global buttonState;
    
    if buttonState.equalize
        % 如果已经启用，关闭该功能并清空右边图像
        buttonState.equalize = false;
        imshow([], 'Parent', ax2);
    else
        % 启用功能
        buttonState.equalize = true;
        if isempty(img)
            msgbox('No image loaded!');
            return;
        end
        gray_img = rgb2gray(img);
        eq_img = histeq(gray_img);
        imshow(eq_img, 'Parent', ax2);  % 显示处理后的图像
    end
end

% 直方图匹配功能
function toggleHistogramMatch(ax1, ax2)
    global img;
    global buttonState;
    
    if buttonState.histogramMatch
        % 如果已经启用，关闭该功能并清空右边图像
        buttonState.histogramMatch = false;
        imshow([], 'Parent', ax2);
    else
        % 启用功能
        buttonState.histogramMatch = true;
        if isempty(img)
            msgbox('No image loaded!');
            return;
        end
        
        % 选择目标图像进行直方图匹配
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp','Image Files'}, 'Select a Target Image');
        if isequal(file, 0)
            return;
        end
        target_img = imread(fullfile(path, file));
        
        % 将源图像转换为灰度图像
        gray_img = rgb2gray(img);
        target_gray_img = rgb2gray(target_img);
        
        % 执行直方图匹配
        matched_img = imhistmatch(gray_img, target_gray_img);
        
        % 显示匹配后的图像
        imshow(matched_img, 'Parent', ax2);  % 显示处理后的图像
    end
end

% 对比度增强功能
function toggleContrastEnhance(ax1, ax2)
    % 弹出对比度增强选项窗口
    dlg = uifigure('Name', '对比度增强', 'Position', [200, 200, 300, 200]);
    
    % 创建“线性变换”按钮
    uibutton(dlg, 'Text', '线性变换', 'Position', [50, 120, 200, 40], ...
             'ButtonPushedFcn', @(btn, event) linearContrastEnhance(ax1, ax2, dlg));
    
    % 创建“非线性变换”按钮
    uibutton(dlg, 'Text', '非线性变换', 'Position', [50, 60, 200, 40], ...
             'ButtonPushedFcn', @(btn, event) nonlinearContrastEnhanceDialog(ax1, ax2, dlg));
end

% 线性对比度增强
function linearContrastEnhance(ax1, ax2, dlg)
    global img;
    
    if isempty(img)
        msgbox('No image loaded!');
        return;
    end
    gray_img = rgb2gray(img);
    enhanced_img = imadjust(gray_img, [0.3 0.7], [0 1]);
    imshow(enhanced_img, 'Parent', ax2);  % 显示处理后的图像
    close(dlg);  % 关闭对比度增强选择窗口
end

% 弹出非线性对比度增强窗口
function nonlinearContrastEnhanceDialog(ax1, ax2, dlg)
    % 创建新窗口，包含“对数变换”和“指数变换”按钮
    dlg2 = uifigure('Name', '非线性变换', 'Position', [250, 250, 300, 200]);
    
    % 对数变换按钮
    uibutton(dlg2, 'Text', '对数变换', 'Position', [50, 120, 200, 40], ...
             'ButtonPushedFcn', @(btn, event) logContrastEnhance(ax1, ax2, dlg2));
    
    % 指数变换按钮
    uibutton(dlg2, 'Text', '指数变换', 'Position', [50, 60, 200, 40], ...
             'ButtonPushedFcn', @(btn, event) expContrastEnhance(ax1, ax2, dlg2));
end

% 对数变换
function logContrastEnhance(ax1, ax2, dlg2)
    global img;
    
    if isempty(img)
        msgbox('No image loaded!');
        return;
    end
    gray_img = rgb2gray(img);
    c = 1;  % 对数变换常数
    log_img = c * log(1 + double(gray_img));
    imshow(log_img, [], 'Parent', ax2);  % 显示处理后的图像
    close(dlg2);  % 关闭非线性变换窗口
end

% 指数变换
function expContrastEnhance(ax1, ax2, dlg2)
    global img;
    
    if isempty(img)
        msgbox('No image loaded!');
        return;
    end
    gray_img = rgb2gray(img);
    c = 1;  % 指数变换常数
    exp_img = c * (exp(double(gray_img)) - 1);
    imshow(exp_img, [], 'Parent', ax2);  % 显示处理后的图像
    close(dlg2);  % 关闭非线性变换窗口
end

% 图像缩放功能
function toggleZoomImage(ax1, ax2)
    global img;
    global buttonState;
    
    if buttonState.zoomImage
        % 如果已经启用，关闭该功能并清空右边图像
        buttonState.zoomImage = false;
        imshow([], 'Parent', ax2);
    else
        % 启用功能
        buttonState.zoomImage = true;
        if isempty(img)
            msgbox('No image loaded!');
            return;
        end
        % 弹出输入框，要求用户输入缩放倍数
        prompt = {'Enter scaling factor (e.g., 1.5):'};
        dlg_title = 'Zoom Image';
        def = {'1.5'};  % 默认值
        answer = inputdlg(prompt, dlg_title, [1 35], def);
        
        if isempty(answer)
            return;
        end
        
        % 获取缩放倍数
        scale = str2double(answer{1});
        
        % 检查输入是否合法
        if isnan(scale) || scale <= 0
            msgbox('Invalid scaling factor! Please enter a positive number.');
            return;
        end
        
        % 进行缩放
        resized_img = imresize(img, scale, 'bilinear');
        imshow(resized_img, 'Parent', ax2);  % 显示处理后的图像
    end
end

% 图像旋转功能
function toggleRotateImage(ax1, ax2)
    global img;
    global buttonState;
    
    if buttonState.rotateImage
        % 如果已经启用，关闭该功能并清空右边图像
        buttonState.rotateImage = false;
        imshow([], 'Parent', ax2);
    else
        % 启用功能
        buttonState.rotateImage = true;
        if isempty(img)
            msgbox('No image loaded!');
            return;
        end
        % 弹出输入框，要求用户输入旋转角度
        prompt = {'Enter rotation angle (degrees):'};
        dlg_title = 'Rotate Image';
        def = {'45'};  % 默认值
        answer = inputdlg(prompt, dlg_title, [1 35], def);
        
        if isempty(answer)
            return;
        end
        
        % 获取旋转角度
        angle = str2double(answer{1});
        
        % 检查输入是否合法
        if isnan(angle)
            msgbox('Invalid angle! Please enter a valid number.');
            return;
        end
        
        % 进行旋转
        rotated_img = imrotate(img, angle, 'bilinear');
        imshow(rotated_img, 'Parent', ax2);  % 显示处理后的图像
    end
end

%添加噪声功能
function toggleAddNoise(ax1, ax2, noise_info)
    global img;
    global buttonState;
    global noisyImg;  % 新增全局变量，用于存储加噪后的图像
    
    if buttonState.addNoise
        % 如果已启用加噪功能，清空右边图像
        buttonState.addNoise = false;
        imshow([], 'Parent', ax2);  % 清空右侧图像
        noise_info.Text = '';  % 清空噪声信息
    else
        % 启用加噪功能
        buttonState.addNoise = true;
        
        if isempty(img)
            msgbox('Please load an image first.');
            return;
        end
        
        % 弹出选择噪声类型对话框
        choice = questdlg('Choose noise type:', 'Noise Selection', 'Gaussian', 'Salt & Pepper', 'Cancel', 'Gaussian');
        
        if strcmp(choice, 'Cancel')
            return;
        end
        
        % 根据选择的噪声类型，调用对应的噪声添加函数
        if strcmp(choice, 'Gaussian')
            noisyImg = addGaussianNoise(ax2);  % 将加噪后的图像保存
            noise_info.Text = 'Gaussian Noise';  % 更新噪声信息
        elseif strcmp(choice, 'Salt & Pepper')
            noisyImg = addSaltAndPepperNoise(ax2);  % 将加噪后的图像保存
            noise_info.Text = 'Salt and Pepper Noise';  % 更新噪声信息
        end
    end
end

% 高斯噪声添加函数
function noisyImg = addGaussianNoise(ax2)
    global img;
    if ~isempty(img)
        prompt = {'Enter noise mean:'};
        dlg_title = 'Gaussian Noise Parameters';
        def = {'0'};  % 默认值
        answer = inputdlg(prompt, dlg_title, [1 35], def);
        
        if isempty(answer)
            return;
        end
        
        meanNoise = str2double(answer{1});  % 获取用户输入的噪声均值
        sigma = 0.1;  % 设置标准差
        noisyImg = double(img) + meanNoise + sigma * randn(size(img));  % 转换为double进行加法操作
        noisyImg = uint8(noisyImg);  % 强制转换为uint8，确保图像像素为有效值范围
        
        % 防止像素值超出0-255范围
        noisyImg(noisyImg > 255) = 255;  
        noisyImg(noisyImg < 0) = 0;
        
        % 显示噪声图像
        imshow(noisyImg, 'Parent', ax2);
    end
end

% 椒盐噪声添加函数
function noisyImg = addSaltAndPepperNoise(ax2)
    global img;
    if ~isempty(img)
        prompt = {'Enter noise density (density):'};
        dlg_title = 'Salt and Pepper Noise Parameters';
        def = {'0.02'};  % 默认值
        answer = inputdlg(prompt, dlg_title, [1 35], def);
        
        if isempty(answer)
            return;
        end
        
        density = str2double(answer{1});
        noisyImg = img;
        p = density;  % 控制椒盐噪声的概率（噪声强度）
        numSalt = round(p * numel(img));  % 计算盐噪声的数量
        numPepper = numSalt;  % 胡椒噪声数量与盐噪声相同

        % 随机生成盐噪声（白色）
        saltIndices = randperm(numel(img), numSalt);
        noisyImg(saltIndices) = 255;

        % 随机生成胡椒噪声（黑色）
        pepperIndices = randperm(numel(img), numPepper);
        noisyImg(pepperIndices) = 0;

        % 显示带噪声图像
        imshow(noisyImg, 'Parent', ax2);
    end
end

%滤波处理
function applyFilter(ax1, ax2)
    global noisyImg;  % 使用加噪后的图像
    if isempty(noisyImg)
        msgbox('请先向图像添加噪声！');
        return;
    end
    
    % 如果图像是彩色图像，则转换为灰度图像
    if size(noisyImg, 3) == 3
        img_gray = rgb2gray(noisyImg);
    else
        img_gray = noisyImg;
    end

    % 弹出选择滤波类型（空域或频域）
    choice = questdlg('请选择滤波领域:', '滤波领域选择', '空域滤波', '频域滤波', '取消', '空域滤波');
    
    if strcmp(choice, '取消')
        return;
    end
    
    switch choice
        case '空域滤波'
            % 弹出选择空域滤波类型
            filterChoice = questdlg('请选择空域滤波类型:', '空域滤波选择', ...
                                    '均值滤波', '中值滤波', '取消', '均值滤波');
            if strcmp(filterChoice, '取消')
                return;
            end
            
            switch filterChoice
                case '均值滤波'
                    filteredImg = applyMeanFilter(img_gray);  % 使用均值滤波
                case '中值滤波'
                    filteredImg = applyMedianFilter(img_gray);  % 使用中值滤波
            end
            
        case '频域滤波'
            % 频域滤波：选择低通滤波或高通滤波
            filterChoice = questdlg('请选择频域滤波类型:', '频域滤波选择', ...
                                    '低通滤波', '高通滤波', '取消', '低通滤波');
            if strcmp(filterChoice, '取消')
                return;
            end
            
            switch filterChoice
                case '低通滤波'
                    filteredImg = applyLowPassFilter(img_gray);  % 低通滤波
                case '高通滤波'
                    filteredImg = applyHighPassFilter(img_gray);  % 高通滤波
            end
    end
    
    % 显示滤波后的图像
    imshow(filteredImg, 'Parent', ax2);

    % 显示“处理成功”的消息框
    msgbox('处理成功', '成功', 'help');  % 弹出消息框
end

% 空域滤波
% 均值滤波
function filteredImg = applyMeanFilter(img)
    [M, N] = size(img);
    filteredImg = img;  % 初始化输出图像
    
    % 使用3x3窗口进行滑动
    for i = 2:M-1
        for j = 2:N-1
            % 获取3x3邻域的像素值
            window = img(i-1:i+1, j-1:j+1);
            % 计算邻域内的均值
            filteredImg(i,j) = mean(window(:));  % 使用均值滤波
        end
    end
end

% 中值滤波
function filteredImg = applyMedianFilter(img)
    [M, N] = size(img);
    filteredImg = img;  % 初始化输出图像
    
    % 使用3x3窗口进行滑动
    for i = 2:M-1
        for j = 2:N-1
            % 获取3x3邻域的像素值
            window = img(i-1:i+1, j-1:j+1);
            % 对邻域中的像素进行排序，取中位数
            filteredImg(i,j) = median(window(:));  % 使用中值滤波
        end
    end
end

% 频域滤波
% 低通滤波
function filteredImg = applyLowPassFilter(img)
    % 对图像进行傅里叶变换
    F = fftshift(myFFT2(double(img)));  % 自定义傅里叶变换并移频
    
    % 获取图像大小
    [M, N] = size(img);
    
    % 创建一个低通滤波器（圆形，中心为低频）
    D0 = 30;  % 截止频率
    [u, v] = meshgrid(-floor(N/2):floor(N/2)-1, -floor(M/2):floor(M/2)-1);  % 频率网格
    D = sqrt(u.^2 + v.^2);  % 频率距离
    
    % 创建低通滤波器 H
    H = double(D <= D0);  % 截止频率范围内为1，其他为0
    
    % 滤波操作
    F_filtered = F .* H;  % 在频域进行滤波
    
    % 逆傅里叶变换，得到滤波后的图像
    filteredImg = real(myIFFT2(ifftshift(F_filtered)));  % 自定义逆傅里叶变换并反移频
    filteredImg = uint8(filteredImg);  % 转换为无符号整数图像
end

%高通滤波
function filteredImg = applyHighPassFilter(img)
    % 获取图像的尺寸
    [M, N] = size(img);
    
    % 将图像转换为 double 类型（如果是 uint8）
    img = double(img);
    
    % 执行傅里叶变换
    imgFFT = fft2(img);
    
    % 创建高通滤波器的掩码
    cutoff = 30; % 截止频率
    [U, V] = meshgrid(1:N, 1:M);
    D = sqrt((U - N / 2).^2 + (V - M / 2).^2);  % 计算到频域中心的距离
    H = double(D > cutoff); % 创建高通滤波器（D > cutoff时为1，否则为0）
    
    % 将滤波器应用到傅里叶变换后的图像
    imgFFTShifted = fftshift(imgFFT);  % 将频域中心移到图像的中心
    filteredFFT = imgFFTShifted .* H;  % 频域滤波
    filteredImgFFT = ifftshift(filteredFFT);  % 将频域图像移回原始位置
    
    % 执行反傅里叶变换，将结果转换回空间域
    filteredImg = real(ifft2(filteredImgFFT));  % 使用real()避免虚部残留
    
    % 将输出图像转换回 uint8 类型（如果需要）
    filteredImg = uint8(filteredImg);
end



% 2D傅里叶变换
function F = myFFT2(img)
    [M, N] = size(img);
    F = zeros(M, N);  % 初始化傅里叶变换结果
    
    % 对每一列进行傅里叶变换
    for x = 1:N
        F(:, x) = fft(img(:, x));
    end
    
    % 对每一行进行傅里叶变换
    for y = 1:M
        F(y, :) = fft(F(y, :));
    end
end

% 2D逆傅里叶变换
function img = myIFFT2(F)
    [M, N] = size(F);
    img = zeros(M, N);  % 初始化逆傅里叶变换结果
    
    % 对每一列进行逆傅里叶变换
    for x = 1:N
        img(:, x) = ifft(F(:, x));
    end
    
    % 对每一行进行逆傅里叶变换
    for y = 1:M
        img(y, :) = ifft(img(y, :));
    end
end

%边缘检测功能
function toggleEdgeDetection(ax1, ax2)
    global img;
    global buttonState;

    if buttonState.edgeDetection
        % 如果已经启用，关闭该功能并清空右边图像
        buttonState.edgeDetection = false;
        imshow([], 'Parent', ax2);
    else
        % 启用功能
        buttonState.edgeDetection = true;
        if isempty(img)
            msgbox('No image loaded!');
            return;
        end
        
        % 弹出选择边缘检测算子的自定义对话框
        choice = customEdgeDetectionDialog();
        
        % 判断用户是否取消选择
        if strcmp(choice, '取消')
            return;
        end
        
        % 转为灰度图像
        gray_img = rgb2gray(img);

        % 根据选择的算子进行边缘检测
        switch choice
            case 'Roberts'
                edge_img = myRobertsEdge(gray_img);
            case 'Prewitt'
                edge_img = myPrewittEdge(gray_img);
            case 'Sobel'
                edge_img = mySobelEdge(gray_img);
            case 'Laplacian'
                edge_img = myLaplacianEdge(gray_img);
        end

        % 在原图旁边显示边缘检测结果
        imshowpair(img, edge_img, 'montage', 'Parent', ax2);
    end
end

% 边缘检测算子选择对话框
function choice = customEdgeDetectionDialog()
    % 创建自定义对话框，窗口大小可以根据需求进行调整
    f = figure('Position', [400, 400, 300, 200], 'Name', '边缘检测选择', 'MenuBar', 'none', 'NumberTitle', 'off');
    
    % 提示信息
    uicontrol('Style', 'text', 'Position', [30, 140, 240, 30], 'String', '请选择边缘检测算子:', 'FontSize', 12);

    % 按钮
    uicontrol('Style', 'pushbutton', 'String', 'Roberts', 'Position', [30, 100, 80, 30], 'Callback', @(src, event) assignChoice('Roberts'));
    uicontrol('Style', 'pushbutton', 'String', 'Prewitt', 'Position', [120, 100, 80, 30], 'Callback', @(src, event) assignChoice('Prewitt'));
    uicontrol('Style', 'pushbutton', 'String', 'Sobel', 'Position', [30, 60, 80, 30], 'Callback', @(src, event) assignChoice('Sobel'));
    uicontrol('Style', 'pushbutton', 'String', 'Laplacian', 'Position', [120, 60, 80, 30], 'Callback', @(src, event) assignChoice('Laplacian'));
    uicontrol('Style', 'pushbutton', 'String', '取消', 'Position', [120, 20, 80, 30], 'Callback', @(src, event) assignChoice('取消'));

    % 变量来存储用户选择的选项
    choice = '';
    
    % 将用户选择的值分配给 choice
    function assignChoice(chosen)
        choice = chosen;
        close(f);  % 关闭对话框
    end

    % 等待用户选择
    uiwait(f);
    
    % 返回用户的选择
    return;
end

% Roberts 边缘检测
function edge_img = myRobertsEdge(img)
    [M, N] = size(img);
    edge_img = zeros(M, N);

    % Roberts 算子的模板
    Gx = [1, 0; 0, -1];
    Gy = [0, 1; -1, 0];

    for i = 1:M-1
        for j = 1:N-1
            % 计算 x 和 y 方向的梯度
            Ix = sum(sum(double(img(i:i+1, j:j+1)) .* Gx));
            Iy = sum(sum(double(img(i:i+1, j:j+1)) .* Gy));
            
            % 计算梯度的幅值
            edge_img(i, j) = sqrt(Ix^2 + Iy^2);
        end
    end
end

% Prewitt 边缘检测
function edge_img = myPrewittEdge(img)
    [M, N] = size(img);
    edge_img = zeros(M, N);

    % Prewitt 算子的模板
    Gx = [-1, 0, 1; -1, 0, 1; -1, 0, 1];
    Gy = [-1, -1, -1; 0, 0, 0; 1, 1, 1];

    for i = 2:M-1
        for j = 2:N-1
            % 计算 x 和 y 方向的梯度
            Ix = sum(sum(double(img(i-1:i+1, j-1:j+1)) .* Gx));
            Iy = sum(sum(double(img(i-1:i+1, j-1:j+1)) .* Gy));
            
            % 计算梯度的幅值
            edge_img(i, j) = sqrt(Ix^2 + Iy^2);
        end
    end
end

% Sobel 边缘检测
function edge_img = mySobelEdge(img)
    [M, N] = size(img);
    edge_img = zeros(M, N);

    % Sobel 算子的模板
    Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
    Gy = [-1, -2, -1; 0, 0, 0; 1, 2, 1];

    for i = 2:M-1
        for j = 2:N-1
            % 计算 x 和 y 方向的梯度
            Ix = sum(sum(double(img(i-1:i+1, j-1:j+1)) .* Gx));
            Iy = sum(sum(double(img(i-1:i+1, j-1:j+1)) .* Gy));
            
            % 计算梯度的幅值
            edge_img(i, j) = sqrt(Ix^2 + Iy^2);
        end
    end
end

% Laplacian 边缘检测
function edge_img = myLaplacianEdge(img)
    [M, N] = size(img);
    edge_img = zeros(M, N);

    % Laplacian 算子的模板
    kernel = [0, 1, 0; 1, -4, 1; 0, 1, 0];

    for i = 2:M-1
        for j = 2:N-1
            % 对每个像素周围进行卷积
            region = double(img(i-1:i+1, j-1:j+1));
            edge_img(i, j) = sum(sum(region .* kernel));
        end
    end
end







% 特征提取函数
% 特征提取函数
function toggleFeatureExtraction(ax1, ax2)
    global img;
    global buttonState;

    % 检查图像是否加载
    if isempty(img)
        msgbox('No image loaded!');
        return;
    end

    % 启动或关闭特征提取功能
    if buttonState.featureExtraction
        % 如果已经启用，关闭该功能并清空右边图像
        buttonState.featureExtraction = false;
        imshow([], 'Parent', ax2);
    else
        % 启用特征提取
        buttonState.featureExtraction = true;

        % 获取阈值分割结果
        threshold = 100;  % 设置一个阈值，适当调整
        bin_img = thresholdSegmentation(img, threshold);

        % 分别显示原始图像和阈值分割后的目标图像
        imshow(img, 'Parent', ax1);
        title(ax1, 'Original Image');

        imshow(bin_img, 'Parent', ax2);
        title(ax2, 'Thresholded Target');

        % 弹出窗口，选择HOG或LBP
        choice = questdlg('Select feature extraction method:', ...
            'Feature Extraction', ...
            'HOG', 'LBP', 'Cancel', 'HOG'); % 默认选HOG

        % 根据选择执行相应的特征提取
        if strcmp(choice, 'HOG')
            % 进行HOG特征提取
            gray_img = rgb2gray(img);  % 将图像转换为灰度图
            hog_features_image = extractHOGAndVisualize(gray_img, ax2);  % 提取并显示HOG特征
        elseif strcmp(choice, 'LBP')
            % 进行LBP特征提取
            gray_img = rgb2gray(img);  % 将图像转换为灰度图
            lbp_features = extractLBPFeatures(gray_img, ax2);  % 提取并显示LBP特征
        end
    end
end

% 自定义阈值分割函数
function bin_img = thresholdSegmentation(img, threshold)
    % 将图像转换为灰度图像
    gray_img = rgb2gray(img);
    
    % 应用阈值分割
    bin_img = gray_img > threshold;  % 大于阈值的部分设为 1，小于阈值的部分设为 0
end

% 自定义的 LBP 特征提取函数
function lbp_features = extractLBPFeatures(img, ax2)
    % 获取图像大小
    [M, N] = size(img);
    
    % 初始化 LBP 特征
    lbp_features = zeros(M-2, N-2);
    
    % 遍历图像的每个像素
    for i = 2:M-1
        for j = 2:N-1
            % 获取 3x3 邻域
            neighbors = img(i-1:i+1, j-1:j+1);
            center = neighbors(2,2);  % 中心像素值
            % 将邻域的每个像素与中心像素进行比较，生成二进制值
            lbp_value = (neighbors > center);
            % 将二进制值转换为十进制数作为LBP值
            lbp_features(i-1, j-1) = bin2dec(num2str(lbp_value(:)'));
        end
    end

    % 显示LBP特征图到ax2
    imshow(lbp_features, [], 'Parent', ax2);  % 显示LBP特征图
    title(ax2, 'LBP Features');
end

function hog_features_image = extractHOGAndVisualize(img, ax2)
    % 检查图像是否是灰度图像
    if size(img, 3) == 3
        img = rgb2gray(img);  % 如果是RGB图像，转换为灰度图
    end
    
    % 计算图像的梯度（水平和垂直方向）
    [Gx, Gy] = gradient(double(img));  % 计算水平和垂直方向的梯度
    magnitude = sqrt(Gx.^2 + Gy.^2);   % 计算每个像素点的梯度幅值
    angle = atan2(Gy, Gx);              % 计算每个像素点的梯度方向

    % HOG特征的参数设置
    num_bins = 9;             % 每个cell内的方向直方图分成9个bin（通常为0到180度之间）
    bin_width = pi / num_bins; % 每个bin的角度宽度（pi/9）
    
    % 设置cell的大小（8x8像素）和block的大小（2x2个cell）
    cell_size = [8, 8];  % 每个cell的大小（8x8像素）
    block_size = [2, 2]; % 每个block包含2x2个cell

    % 获取图像的大小
    [M, N] = size(img);
    
    % 计算HOG特征
    hog_features = [];
    
    % 创建一个图像来表示HOG特征的可视化
    hog_features_image = zeros(M, N);
    
    % 遍历图像的每个cell，并计算每个cell的方向直方图
    for i = 1:cell_size(1):M - cell_size(1) + 1
        for j = 1:cell_size(2):N - cell_size(2) + 1
            % 提取当前cell区域的梯度幅值和方向
            cell_magnitude = magnitude(i:i+cell_size(1)-1, j:j+cell_size(2)-1);
            cell_angle = angle(i:i+cell_size(1)-1, j:j+cell_size(2)-1);
            
            % 初始化cell的方向直方图
            cell_hist = zeros(1, num_bins);
            
            % 对cell内每个像素计算方向直方图
            for m = 1:cell_size(1)
                for n = 1:cell_size(2)
                    % 计算每个像素的方向和幅值
                    bin_idx = floor(mod(cell_angle(m, n), 2*pi) / bin_width) + 1;
                    if bin_idx > num_bins
                        bin_idx = 1;  % 防止超出最大索引
                    end
                    % 将幅值加到对应的方向bin中
                    cell_hist(bin_idx) = cell_hist(bin_idx) + cell_magnitude(m, n);
                end
            end
            
            % 计算每个cell的主要方向（即最大值的方向）
            [~, max_bin_idx] = max(cell_hist);
            main_angle = (max_bin_idx - 1) * bin_width;  % 主要方向的角度
            
            % 在 hog_features_image 中为该 cell 区域填充梯度方向
            % 使用该 cell 中的主要方向显示箭头，强度代表幅值
            for m = 1:cell_size(1)
                for n = 1:cell_size(2)
                    % 计算当前像素的方向，并计算对应的梯度幅值
                    pixel_angle = angle(i+m-1, j+n-1);
                    pixel_magnitude = magnitude(i+m-1, j+n-1);
                    
                    % 将梯度的方向和强度映射到hog_features_image中
                    hog_features_image(i+m-1, j+n-1) = pixel_magnitude * cos(pixel_angle - main_angle);
                end
            end
        end
    end
    
    % 对HOG特征图进行归一化以增强对比度
    hog_features_image = mat2gray(hog_features_image);  % 将HOG特征图归一化到[0, 1]范围

    % 可以选择进一步增强对比度
    % hog_features_image = imadjust(hog_features_image);

    % 显示HOG特征图到ax2
    imshow(hog_features_image, [], 'Parent', ax2);
    title(ax2, 'HOG Features');
end
