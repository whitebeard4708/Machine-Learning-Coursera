function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
test_value = [0.01 0.03 0.1 0.3 1 3 10 30]';
m = length(test_value);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_temp = 0.01;
sigma_temp = 0.01;
lowest_cost = 10000;
for i = 1:m
  C_temp = test_value(i)
  for j = 1:m
    sigma_temp = test_value(j)
    model1 = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    prediction_temp = svmPredict(model1, Xval);
    cost = mean(double(prediction_temp ~= yval));
    if cost < lowest_cost
      C = C_temp
      sigma = sigma_temp
      lowest_cost = cost;
    endif
  endfor
endfor




% =========================================================================

end
