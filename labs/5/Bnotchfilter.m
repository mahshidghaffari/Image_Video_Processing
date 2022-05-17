function h = Bnotchfilter(f0, n1, n2, n, d0);

d0 = 1.2*d0;
% Butterworth notch filter
[k1 k2] = meshgrid(-round(n2/2)+1:round(n2/2), -round(n1/2)+1:round(n1/2));
d1 = sqrt((k1-f0).^2 + (k2-f0).^2);
d2 = sqrt((k1+f0).^2 + (k2+f0).^2);
denh = 1+((d0^2)./(d1.*d2)).^n;

h = zeros(n1,n2);
h = 1./denh;

%figure;imshow(h)

