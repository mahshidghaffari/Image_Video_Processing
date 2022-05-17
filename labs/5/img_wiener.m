%Wiener filter for only additive noise
close all;clear all;clc

x = double(rgb2gray(imread('redfruits.jpg')));
[n1 n2] = size(x);

f = fftshift(fft2(x));
sf = abs(f).^2;

% Add noise that follows a normal distribution
nm = 130; 
n = nm*randn(size(x));

fn = fftshift(fft2(n));
sn = abs(fn).^2;

y = x + n;

fy = fftshift(fft2(y));

% Assume snr is known:
snr = sn./sf;
fest = (fy)./(1+snr);

xest = ifft2(fest);

figure;imshow(x/255)
figure;imshow(y/255)
figure;imshow(abs(xest)/255)

disp 'error before'
mean(mean((y)-abs(x)))
disp 'error after'
mean(mean((y)-abs(xest)))

% 
% figure;
% subplot(1,3,1);imshow(abs(x)/255)
% subplot(1,3,2);imshow(abs(y)/255)
% subplot(1,3,3);imshow(abs(xest)/255)
% 


