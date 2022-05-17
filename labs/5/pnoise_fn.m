function n = pnoise_fn(f0, n1, n2)
% close all;clear all;clc

[t1, t2] = meshgrid(1:n2,1:n1);
%n = sin(2*pi*f0*((t1-1)/n2 + (t2-1)/n1));
n = sin(2*pi*f0*((t1-1)/n2));

%figure;imagesc(n)

fn = fftshift(fft2(n));

% figure;imagesc(abs(fn));
% figure;mesh(abs(fn))
% figure;plot(abs(fn));