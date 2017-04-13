function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute cost = J
for i=1:m
  if (y(i)) % TRUE, 1, etc
    J = J - log(sigmoid(X(i,:) * theta));
  else % FALSE, 0, etc
    J = J - log(1 - sigmoid(X(i,:) * theta));
  end
end
J = J / m;
l = 0;
for i=2:size(theta)
    l = l + theta(i)^2;
end
l = (l * lambda) / (2 * m);
J = J + l;

% Compute Gradient = grad.
grad(1) = 0;
for i=2:size(theta)
  g = 0;
  for k=1:m
    g = g + ((sigmoid(X(k,:) * theta) - y(k)) * X(k,i));
  end
grad(i) = (g + lambda * theta(i)) / m;
end






% =============================================================

end
