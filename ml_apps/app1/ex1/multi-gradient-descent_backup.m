function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

thetaLenght =  size(theta,1);
thetaList = zeros(thetaLenght,1);

for iter = 1:num_iters
  % Save the cost J in every iteration    
      
     
  for thetaItr = 1:thetaLenght
      thetaList(thetaItr) = theta(thetaItr) - alpha / m * sum((X * theta - y) .* X(:, thetaItr));
  end
  
      theta = thetaList;
      % J_history helping me analysising how cost function is reduting as my gradient function
      % changing the value of theta
      J_history(iter) = computeCost(X, y, theta);
end



m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thetas = zeros(size(X, 2), 1);	
for iter = 1:num_iters
	
	% Same computation as gradientDescent.m, but we must loop over all features.
	for i = 1:size(X, 2),
	    t = theta(i) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, i));
		thetas(i) = t;
	end	
	theta = thetas;
end
