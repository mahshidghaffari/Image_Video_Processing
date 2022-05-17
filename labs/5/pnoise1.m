close all;clear all; clc

x = double(rgb2gray(imread('apple.jpg')));
f = fftshift(fft2(x));

[n1,n2] = size(x);

f0 = 40;
n = pnoise_fn(f0, n1, n2);

fn = fftshift(fft2(n));

figure;imagesc(n)

k = 50;
xn = x + k*n;

fxn = fftshift(fft2(xn));

figure;imagesc(log(abs(fxn)))

figure;imshow(xn/255)