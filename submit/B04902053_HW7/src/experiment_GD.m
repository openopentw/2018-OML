FILENAME = 'kddb';

tic;
[y, X] = libsvmread(FILENAME);
toc
disp('Finish loading data.');

% transform y-data from (0, 1) to (-1, 1)
y = y * 2 - 1;

tic;
w = GD(y, X);
toc

%% evaluate the result
C = 1e-1;
err = 1 / 2 * (w' * w) + C * sum(log(1 + exp(- y .* (X * w))));
disp(err);
