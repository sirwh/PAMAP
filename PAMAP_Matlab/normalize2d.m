function output=normalize2d(input)
% normalize a 2d data as (x−μ)/σ , where μ and σ are mean and standard deviation
% mean and standard are operated on COLUMNs

output=(input-repmat(mean(input),size(input,1),1))./repmat(std(input),size(input,1),1);

% e.g. the function normalizes
%    1     2          -1    -1
%    3     4    to    0     0
%    5     6           1     1
