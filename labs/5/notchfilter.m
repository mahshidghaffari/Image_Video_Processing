function h = notchfilter(f0, n1,n2);

% close all; clear all; clc
% n1 = 1080; n2 = 1080; f0 = 80;
[k1 k2] = meshgrid(-round(n2/2)+1:round(n2/2), -round(n1/2)+1:round(n1/2));
d1 = sqrt((k1-f0).^2 + (k2-f0).^2);
d2 = sqrt((k1+f0).^2 + (k2+f0).^2);
d0 = 40;
h = zeros(n1,n2);
h(d1 < d0) = 1;
h(d2 < d0) = 1;

%figure;imshow(h)

