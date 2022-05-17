close all;clear all; clc

x =double(rgb2gray(imread('redfruits.jpg')));
f1 = fftshift(fft2(x));
figure;imagesc(log(abs(f1)))


[n1 n2] = size(x);
f2 = f1; f2(n1/2+1,n2/2+1) = 0;
figure;imagesc(log(abs(f2)))

y = ifft2(f2);

figure;imshow(abs(x)/255)
figure;imshow(abs(y)/255)
