function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

thetaLength =  size(theta,1);
thetaList = zeros(thetaLength,1);

for iter = 1:num_iters
  % Save the cost J in every iteration    
      
     
  for thetaItr = 1:thetaLength
      thetaList(thetaItr) = theta(thetaItr) - alpha / m * sum((X * theta - y) .* X(:, thetaItr));
  end
  
      theta = thetaList;
      % J_history helping me analysising how cost function is reduting as my gradient function
      % changing the value of theta
      J_history(iter) = computeCost(X, y, theta);
end
