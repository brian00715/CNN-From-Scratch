% Description: calculate the numerical gradient of a function J w.r.t. theta
% Input:
%   J: a function handle
%   theta: the parameter vector
% Output:
%   numgrad: the numerical gradient of J w.r.t. theta
function numgrad = calcuGrad(J, theta)
    % This file is modified based on UFLDL Deep Learning Tutorial
    % http://ufldl.stanford.edu/tutorial/

    % Initialize numgrad with zeros
    numgrad = zeros(size(theta));

    epsilon = 1e-4;

    for i = 1:length(numgrad)
        oldT = theta(i);
        theta(i) = oldT + epsilon;
        pos = J(theta);
        theta(i) = oldT - epsilon;
        neg = J(theta);

        if pos * neg < 0
            numgrad(i) = 0;
        else
            numgrad(i) = (pos - neg) / (2 * epsilon);
        end

        theta(i) = oldT;

        if mod(i, 100) == 0
            fprintf('Done with %d\n', i);
        end

    end

end
