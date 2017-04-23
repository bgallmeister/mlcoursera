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

% First do the prediction using the passed-in X and Theta.
X = [ones(m, 1) X];             % Add on bias unit to input.

BigDelta2 = zeros(size(Theta2_grad));
BigDelta1 = zeros(size(Theta1_grad));

partialsum = 0;
for i = 1:m
    % Forward prop for X(i)
    z2 = X(i,:) * Theta1';          % 1 x 25.
    a2 = sigmoid(z2);
    a2 = [1 a2];                    % Add bias unit to hidden layer.
    h = sigmoid(a2 * Theta2');      % 1 x 10.
    [unused, my_y] = max(h, [], 2);           % index of the biggest element.
    
    % Recode My_Y as vector.  I bet there is a spiffy Matlab way to do this.
    %my_ymatrix = zeros(num_labels, 1);
    %my_ymatrix(my_y) = 1;

    % Recode Y as vector.  I bet there is a spiffy Matlab way to do this.
    ymatrix = zeros(1, num_labels);
    ymatrix(y(i)) = 1;
    
    % Forward prop done;  determine cost for this sample.
    % Partial result.
    partial = log(h) .* ymatrix + log(1-h) .* not(ymatrix);
    partialsum = partialsum + sum(partial);
    
    % Backprop.
    LittleDelta3 = (h - ymatrix); 
    LittleDelta2 = LittleDelta3 * Theta2 .* [1 sigmoidGradient(z2)];
    BigDelta2 = BigDelta2 + LittleDelta3' * a2;
    BigDelta1 = BigDelta1 + LittleDelta2(2:hidden_layer_size+1)' * X(i,:);
end


% Now sum it all and divide by m...
% non-regularized:
% J = (-1 * sum(partial)) / m;
% regularized:
% theta(1) = 0;  % Don't regularize theta(0).
% J = ((-1 * sum(partial)) + (lambda * (theta.' * theta) / 2)) / m;
t1 = Theta1(:,2:input_layer_size+1);
t2 = Theta2(:,2:hidden_layer_size+1);
J = ((-1 * partialsum) + ((lambda/2) * (sum(sum(t1 .* t1)) + sum(sum(t2 .* t2))))) / m;


Theta1_grad = (BigDelta1 + lambda * Theta1) / m;
Theta1_grad(:,1) = BigDelta1(:,1) / m;
Theta2_grad = (BigDelta2 + lambda * Theta2) / m;
Theta2_grad(:,1) = BigDelta2(:,1) / m;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
