close all; clear all;

x = rgb2gray(imread('win1.jpg')); x = double((x));
n1 = size(x,1); n2 = size(x,2); 
[k1 k2] = meshgrid(1:n2,1:n1);
[u v] = meshgrid(-1+2/n2:2/n2:1,-1+2/n1:2/n1:1); 
%[u v] = meshgrid(-n2/2+1:n2/2,-n1/2+1:n1/2); 
% u = 2*u/n2;v = 2*v/n1;
whos u v

F = fft2(x);

a = .13; b = 0;

% Blurring Function
H = sinc((u*a + v*b)).*exp(-j*pi*(u*a + v*b)); 
G = F.*H;

% Motion Blurred image 
g = (ifft2(G));
figure;imshow(abs(g)/255)

% Noisy AND Motion Blurred image 
xn = imnoise(uint8(abs(g)),'gaussian',0,.002); 
xn = double(xn);
Fn = fft2(xn);
figure;imshow(xn/255)

% A basic way to assess the power spectrum of the noise, with the
% assumption the original image x and the motion blurred and noisy image xn
% are known. We conisder the pproximation of the noise is x-xn
nn = x-xn;
snn = abs(fft2(nn)).^2;
sxx = abs(fft2(x)).^2;

% 2D Wiener filter
dh = abs(H).^2 + snn./sxx; 
Hw = conj(H)./dh;

% Wiener filtered motion blurred image
 
R1 = Hw.*G;
rx1 = abs(ifft2(R1)); 
figure;imshow(rx1/255);

% Wiener filtered motion blurred and noisy image
R2 = Hw.*Fn;
rx2 = abs(ifft2(R2)); 
figure;imshow(rx2/255);

% Application of an inverse filter to an image with additive noise and with motion blur
R3 = Fn./H;
rx3 = abs(ifft2(R3)); 
figure;imshow(rx3/255);

% Matlab Wiener filter
r0 = wiener2(abs(xn));
figure;imshow(r0/255);


