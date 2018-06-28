FILENAME = 'kddb';

tic;
[y, X] = libsvmread(FILENAME);
toc
disp('Finish loading data.');

% transform y-data from (0, 1) to (-1, 1)
y = y * 2 - 1;

tic;
model = train(y, X, '-s 0');
toc

tic;
[predicted_label, accuracy, decision_values] = predict(y, X, model);
toc

%% evaluate the result
disp(accuracy);
