function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
newtheta = zeros(size(theta,1), 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = zeros(size(X,2),1);
    for j = 1:size(X,2)
        h(j) = 0;
        for row = 1:m
            % Accumulate h(x)
            h(j) = h(j) + ((X(row,:) * theta) - y(row)) * (X(row,j));
        end
        h(j) = (h(j) * alpha) / m;
        newtheta(j) = theta(j) - h(j);
    end
    theta = newtheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    fprintf('Cost at iter %d: %f\n', iter, J_history(iter));

end

end
