% Entrega 2
% G09
%     Diego Sota Rebollo
%     David Santa Cruz Del Moral
% 
% Objetivo Obligatorio %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all

% 1º: Suavizado de la imagen en escala de grises.
    % Se carga la imagen y se pasa a escala de grises.
I = imread("G09.jpg");
img_grey =  rgb2gray(I);

imshow(img_grey);

    % Suavizado de la imagen monocromo mediante un filtro de media.
mascara = (1/(11*11)*ones(11,11));

img_grey_Fmedia = imfilter(img_grey, mascara);

imshow(img_grey_Fmedia);


% 2º: Analizar y comparar el módulo de la FFT de la imagen monocromo y la imagen suavizada.
    %  FFTs.
img_grey_FFT = fftshift(fft2(double(img_grey)));    % Filas y columnas (dimensiones)
FFT_modulo_img_grey = abs(img_grey_FFT);
FFT_fase_img_grey = angle(img_grey_FFT);

img_grey_Fmedia_FFT = fftshift(fft2(double(img_grey_Fmedia)));    
FFT_modulo_img_grey_Fmedia = abs(img_grey_Fmedia_FFT);
FFT_fase_img_grey_Fmedia = angle(img_grey_Fmedia_FFT);

figure;
subplot(1, 2, 1), mesh(FFT_modulo_img_grey), title('Módulo Imagen Monocromo'), xlabel('u'), ylabel('v');
subplot(1, 2, 2), mesh(FFT_modulo_img_grey_Fmedia), title('Módulo Imagen Suavizada'), xlabel('u'), ylabel('v');

figure;
subplot(1, 2, 1), imshow(FFT_modulo_img_grey,[]), title('Módulo Imagen Monocromo')
subplot(1, 2, 2), imshow(FFT_fase_img_grey,[]), title('Fase Imagen Monocromo')

figure;
subplot(1, 2, 1), imshow(FFT_modulo_img_grey_Fmedia,[]), title('Módulo Imagen Suavizada')
subplot(1, 2, 2), imshow(FFT_fase_img_grey_Fmedia,[]), title('Fase Imagen Suavizada')

    % Transformación logarítmica.
FFT_log_img_grey = log10(1 + FFT_modulo_img_grey);
FFT_log_img_grey_Fmedia = log10(1 + FFT_modulo_img_grey_Fmedia);

figure;
subplot(2, 1, 1), imshow(FFT_log_img_grey,[]), title('Modulo con Transformación logarítmica Imagen monocromo');
subplot(2, 1, 2), imshow(FFT_log_img_grey_Fmedia,[]), title('Modulo con Transformación logarítmica Imagen suavizada');


% 3º: Estimación de la PSF en el dominio de la frecuencia.
    % Se aplicará: G(u,v) = F(u,v) * H(u,v) -> H(u,v) = G(u,v) / F(u,v) (asumiendo ruido = 0).
F = img_grey_FFT;
G = img_grey_Fmedia_FFT;     

H = G ./ F;                           

figure;
imshow(log10(abs(H)), []), title('Módulo de la PSF');

% 4º: Restauración de imagen mediante la PSF.
    % F_restaurada(u,v) = R(u,v) * G(u,v)
    % R(u,v) = 1/H(u,v) --> F_restaurada(u,v) = G(u,v)/H(u,v)
R = 1 ./ H;
F_restaurada = R .* G;
F_restaurada_log = log10(1 + F_restaurada);

img_grey_recovered = abs(ifft2(F_restaurada));

figure;
imshow(abs(F_restaurada_log), []), title('Módulo de la imagen restaurada');

figure;
imshow(img_grey_recovered, []), title('Imagen Restaurada'); 

%% 
% Objetivo Creativo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all

% 1º: Extraer componentes RGB
    % Se carga la imagen .
I = imread("G09.jpg");

figure;
imshow(I), title('Imagen RGB');

    % Se extraen las componentes de color
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);


% 2º: Obtener imagen resultante de añadir movimiento en cada componente.
    % Componentes Transformadas.
R_FFT = fft2(double(R), 1706, 2560);   
G_FFT = fft2(double(G), 1706, 2560);
B_FFT = fft2(double(B), 1706, 2560);

    % H para cada componente
lenR = 30; thetaR = -45; lenG = 20; thetaG = -90; lenB = 40; thetaB = -135;

H_R = fft2(fspecial('motion',lenR,thetaR),1706,2560);
H_G = fft2(fspecial('motion',lenG,thetaG),1706,2560);
H_B = fft2(fspecial('motion',lenB,thetaB),1706,2560);

moved_img = I;

    % Producto espacial con movimiento
moved_img(:,:,1) = ifft2(H_R.*R_FFT);
moved_img(:,:,2) = ifft2(H_G.*G_FFT);
moved_img(:,:,3) = ifft2(H_B.*B_FFT);
moved_img = uint8(moved_img);

figure;
subplot(2,1,1), imshow(real(log10(1 + H_R)), []), title('Original Red')
subplot(2,1,2), imshow(real(log10(1 + H_R.*R_FFT)), []), title('len=30, th = -45')

figure;
subplot(2,1,1), imshow(real(log10(1 + H_G)), []), title('Original Green')
subplot(2,1,2), imshow(real(log10(1 + H_G.*G_FFT)), []), title('len=20, th = -90')

figure;
subplot(2,1,1), imshow(real(log10(1 + H_B)), []), title('Original Blue')
subplot(2,1,2), imshow(real(log10(1 + H_B.*B_FFT)), []), title('len=40, th = -135')

figure;
imshow(moved_img, []), title('Color Moved')


% 3º: Restaurar imagen original.
    % Filtros de Wiener
invH_R = (abs(R_FFT).^2) .* conj(H_R)./(abs(R_FFT).^2).* abs(H_R).^2;
invH_G = (abs(G_FFT).^2) .* conj(H_G)./(abs(G_FFT).^2).* abs(H_G).^2;
invH_B = (abs(B_FFT).^2) .* conj(H_B)./(abs(B_FFT).^2).* abs(H_B).^2;

fixed_img = I;

fixed_img(:,:,1) = ifft2(invH_R.*fft2(double(moved_img(:,:,1))));
fixed_img(:,:,2) = ifft2(invH_G.*fft2(double(moved_img(:,:,2))));
fixed_img(:,:,3) = ifft2(invH_B.*fft2(double(moved_img(:,:,3))));

    % Imagen final.
fixed_img = uint8(fixed_img);

figure;
imshow(fixed_img, []), title('Imagen Restaurada')
