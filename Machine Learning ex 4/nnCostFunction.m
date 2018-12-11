function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
  
for i = 1:m
  a1 = [1 X(i,:)]; % a 1x401
  z2 = a1 * Theta1'; % 1x26
  a2 = [1 sigmoid(z2)];
  z3 = a2 * Theta2'; % 1x10
  a3 = sigmoid(z3)'; % 10x1
  ans = zeros(num_labels, 1); % recorded vector
  ans(y(i)) = 1;
  J = J + (-1/m) * (ans'*log(a3) + (1-ans)'*log(1 - a3));
endfor

% gradient
Theta1_noBias= Theta1(:, 2:end); % omit the bias unit
Theta2_noBias= Theta2(:, 2:end); % omit the bias unit

J = J + (lambda/(2*m)) * (sum(sum(Theta1_noBias.*Theta1_noBias)) ...
            + sum(sum(Theta2_noBias.*Theta2_noBias)))

delta_1 = 0;
delta_2 = 0;

for t = 1:m
  % Step 1: set input values
  a1 = [1; X(t,:)']; % 401 x 1
  z2 = Theta1 * a1; % 25 x 1
  a2 = [1; sigmoid(z2)]; % 26 x 1, add bias unit
  z3 = Theta2 * a2; 
  a3 = sigmoid(z3); % 10 x 1, add bias unit
  
  % Step 2: find error3: output layer
  correct = zeros(num_labels, 1);
  correct(y(t)) = 1;
  error3 = a3 - correct;   % vector Kx1;
  
  % Step 3: find error2: hidden layer
  error2 = (Theta2_noBias' * error3).* sigmoidGradient(z2); % vector hidden layer x 1
  
  % Step 4: accumulate
  delta_2 += error3 * a2';
  delta_1 += error2 * a1';
endfor

% Step 5: Obtain the unregularized gradient
Theta1_grad = (1/m) * delta_1;
Theta2_grad = (1/m) * delta_2;

% Add the regularized parts
Theta1_grad(:, 2:end) += (lambda/m) * Theta1_noBias;
Theta2_grad(:, 2:end) += (lambda/m) * Theta2_noBias;

grad = [Theta1_grad(:) ; Theta2_grad(:)];












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
