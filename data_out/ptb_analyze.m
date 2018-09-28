%% analyze ptb outputs:
clr;
load('ssvep_128_annotate\ssvep_128_annotate.mat');
idx = 1;
if exist('x_val', 'var')
    samples = size(x_val, 1);
    figure(1);
    for i = 1:samples
        fprintf('Sample #%d \n', i);
        total = sum(squeeze(y_val(i, :)));
        total2 = sum(squeeze(y_out(i, :)));
        if total(1)
            subplot(4, 1, 1); plot(squeeze(x_val(i, :, :)));
            subplot(4, 1, 2); plot(squeeze(y_prob(i, :, :)));
            subplot(4, 1, 3); plot(squeeze(y_out(i, :, :)));
            subplot(4, 1, 4); plot(squeeze(y_val(i, :, :))); title('true vals');
            aaaaa = input('Continue? \n');
        end
    end
end