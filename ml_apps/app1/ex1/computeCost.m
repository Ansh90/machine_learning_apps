function J = computeCost(X, y, theta)
J = 0;
% X = [i=97 j=2] , theta = [i =2, j =1]
hypothesisMatrix = X * theta;  % step 1 produce m*1 matrix for any given theta
% h(x) = sum of *(thetas * all the features of the training element ie. 1 row)
error = hypothesisMatrix - y;
errorSquare = error .^2;
J = 1/(2 * length(y)) * sum(errorSquare);
end


