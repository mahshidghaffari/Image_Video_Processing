close all; clear all; clc

x = double(rgb2gray(imread('music.jpg')));

figure;imshow(x/255)

[n1, n2] = size(x);
% You can choose any number m for noise magnitude. 
% For noise proportional to the variations in the image texture, you can use m = std(x(:));
m = 250;

f0 = 80; 
n = m*pnoise(f0, n1, n2);

y = x+n;

figure;imshow(y/255)

fx = fftshift(fft2(x));
fy = fftshift(fft2(y));
fn = fftshift(fft2(n));

% figure;imshow(log(abs(fx))/20);
% figure;imshow(log(abs(fy))/20);
% 
% figure;plot(log(abs(fx(n1/2+1,:)))/50);
% figure;plot(log(abs(fy(n1/2+1,:)))/50);
% figure;plot(log(abs(fn(n1/2+1,:))));

d0 = 40; nb = 2;
h1 = notchfilter(f0, n1,n2);
h2 = Bnotchfilter(f0, n1,n2, nb, d0);

fz1 = fy.*(1-h1);
z1 = ifft2(fz1);

fz2 = fy.*(h2);
z2 = ifft2(fz2);

figure;imshow(abs(z1)/255)
figure;imshow(abs(z2)/255)
