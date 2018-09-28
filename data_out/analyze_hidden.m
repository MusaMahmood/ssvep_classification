%% INCART Dataset - Using transformative annotation method:
% Load Data:
clr;
% load('ssvep_128_annotate\hidden\h_ssvep_128_annotate.mat');
% load('ssvep_256_annotate\hidden\h_ssvep_256_annotate.mat');
load('ssvep_512_annotate\hidden\h_ssvep_512_annotate.mat');
l_titles = {'input', 'conv1d_1', 'conv1d_2', 'flatten_1', 'dense_1', 'dense_2', 'y_true'};
layers = {inputs, conv1d_1, conv1d_2, flatten_1, dense_1, dense_2, y_true};
clear conv1d_1 conv1d_2 flatten_1 y_true dense_1 dense_2
samples = size(inputs, 1);
for s = 1:samples
    figure(1); subplot(2, 1, 1); plot(squeeze(layers{1}(s, :, 1))); title([l_titles{1} '1']); xlim([0, length(layers{1}(s, :, 1))]);
    subplot(2, 1, 2); plot(squeeze(layers{1}(s, :, 2))); title([l_titles{1} '2']); xlim([0, length(layers{1}(s, :, 1))]);
    for i = 1:length(layers)-1
        figure(2); subplot(3, 2, i);
        imagesc(squeeze(layers{i}(s, :, :))); title(l_titles{i});
        colorbar; colormap(jet);
    end
    in = input('Continue? \n');
end
