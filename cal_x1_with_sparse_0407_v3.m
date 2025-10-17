clc; clear; close all;

%% === 初始化参数 ===
size1 = 129; size2 = 129; size3 = 5;
size12 = size1 * size2;
size123 = size12 * size3;

%% === 创建保存路径 === 
% adr_o = "F:\cjx\data\20250102\0408\scatter\red\1005\CS_90\b_XX\100_100_0.3\";
% adr_o = "F:\cjx\data\20250102\0408\scatter\blue\2200ms\1005\CS_90\b_XX\100_100_0.6\";
% adr_o = "F:\cjx\data\20250102\0408\scatter\red\1005\CS_0\0.001_1\";
% adr_o = "F:\cjx\data\20250102\0408\scatter\green\1005\CS_90\b_XX\10_100_0.6\";
adr_o = "F:\cjx\data\20250102\0408\1007\scatter_0.05\green\3000ms\CS_90\0.2\";

mkdir(adr_o);
fid = fopen(fullfile(adr_o, 'idx_info.txt'), 'w');

%% === 加载并预处理全量 R0（只做一次） ===
% R0_full = h5read("F:\cjx\data\20250102\on_center\speckle_red\cljz_npy\32_300_300_90\cljz_norm_float32.h5", '/R0')';
% R0_full = h5read("F:\cjx\data\20250102\on_center\speckle\cljz_npy\blue\32_300_300_0\cljz_norm_float32.h5", '/R0')';
R0_full = h5read("F:\cjx\data\20250102\on_center\speckle\cljz_npy\green\32_300_300_90\cljz_norm_float32.h5", '/R0')';
R_mean = mean(R0_full(:));
R0_full = R0_full / R_mean;
mean_R1 = mean(R0_full, 1);
for i = 1:size(R0_full, 2)
    R0_full(:, i) = R0_full(:, i) - mean_R1(i);
end

%% === 图像数据导入路径 === 
Path = "F:\cjx\data\20250102\0408\1007\scatter_0.05\green\3000ms\y_90\"; 
% Path = "F:\cjx\data\20250102\0408\scatter\red\1005\y_90\"; 
% Path = "F:\cjx\data\20250102\0408\scatter\blue\2200ms\1005\y_90\";
% Path = "F:\cjx\data\20250102\0408\scatter\green\1005\y_90\";

File = dir(fullfile(Path, '*.tiff'));
Tif_num = length(File);
tic;

%% === 主循环处理每张图像 === 
for k = 1:Tif_num
    % 加载图像
    tif_name = File(k).name;
    [~, name, ~] = fileparts(tif_name);
    y = reshape(double(imread(fullfile(Path, tif_name)))', 300 * 300, 1);

    % 阈值选点
    u = 0.5;
    mean_val = mean(y); max_val = max(y);
    threshold1 = mean_val + u * (max_val - mean_val);
    threshold2 = mean_val - u * (max_val - mean_val);
    idx = sort([find(y > threshold1); find(y < threshold2)]);

    % 保存 idx 信息
    fprintf(fid, '%s\t%d\n', name, length(idx));

    % y 预处理
    y = y(idx);
% %    % === 创建独立图窗口并绘图 ===
% %     fig = figure('Visible', 'off');  % 创建不可见图窗，避免弹窗占资源
% %     plot(y);
% %     title(['y Data - ', name], 'Interpreter', 'none');
% %     xlabel('Index');
% %     ylabel('Intensity');
% %     
% %     % === 保存图像（不被覆盖）===
% %     plot_filename = fullfile(adr_o, strcat(name, '_y_plot.png'));
% %     saveas(fig, plot_filename);  % 每张图独立保存
% %     close(fig);  % 关闭图窗，释放内存资源
% %     
% %     % === 保存 y 数据为 .txt 文件 ===
% %     txt_filename = fullfile(adr_o, strcat(name, '_y_data.txt'));
% %     writematrix(y, txt_filename, 'Delimiter', 'tab');

    y_mean = mean(y);
    y = y - y_mean;

    % 提取对应 R0 子矩阵（已标准化）
    R0 = R0_full(idx, :);
%     R0 = R0_full;

    % === 重建 ===
    for mu1 = 0  %tv系数,可修改
        for mu2 = 1  %秩约束项系数,可修改
            num_irls = 2;
            num_grade = 10;
            filename_mu = strcat('_mu1_', num2str(mu1), '_mu2_', num2str(mu2));
            filename_xre = strcat(adr_o, name, filename_mu);

            xre = recon_sparse_rank(filename_xre, y, R0, mu1, mu2, ...
                                    num_irls, num_grade, size1, size2, size3);
            xre = xre * y_mean / R_mean;

            % 拼接多光谱图像
            for i = 1:size3
                object_xre(1:size2, (i-1)*size1+1:i*size1) = ...
                    reshape(xre((i-1)*size12+1:i*size12), size1, size2);
            end
            toc
            % 保存图像
            % 获取当前日期字符串（格式：yyyyMMdd）
            today_str = string(datetime("today", "Format", "yyyyMMdd"));
            filename0 = strcat(name, '_steps=', num2str(num_irls*num_grade), ...
                   filename_mu, '_', today_str);
            imwrite(uint8(object_xre / max(object_xre(:)) * 255), ...
                    strcat(adr_o, 'object_xre_', filename0, '.tiff'));
        end
    end
end

fclose(fid);
toc;
