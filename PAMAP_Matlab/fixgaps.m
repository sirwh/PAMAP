function y=fixgaps(x);
% FIXGAPS Linearly interpolates gaps in a time series
% YOUT=FIXGAPS(YIN) linearly interpolates over NaN
% in the input time series (may be complex), but ignores
% trailing and leading NaN.
%

% R. Pawlowicz 6/Nov/99

y=x;

bd=isnan(x);
gd=find(~bd);

bd([1:(min(gd)-1) (max(gd)+1):end])=0;


y(bd)=interp1(gd,x(gd),find(bd));

% e.g. fixgaps([1 NaN 2 NaN NaN 4]) will output      1.0000    1.5000    2.0000    2.6667    3.3333    4.0000