% Entrega 1
% G09
%     Diego Sota Rebollo
%     David Santa Cruz Del Moral
% 
% Objetivo Obligatorio %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear all, close all

% Se carga la imagen. Se almacenan algunos datos útiles y se carga la
% imagen.

image = imread("G09.jpg");
img_size = size(image);

figure;
subplot(1, 1, 1), imshow(image), title('Imagen True Color');

figure
subplot(3, 1, 1),imhist(image(:, :, 1)), axis auto, title('Rojo')
subplot(3, 1, 2),imhist(image(:, :, 2)), axis auto, title('Verde')
subplot(3, 1, 3),imhist(image(:, :, 3)), axis auto, title('Azul')

% Se extrae la componente verde, su histograma y el número de elementos de
% este.
green = image(:, :, 2);
g_hist = imhist(green);
d_range = numel(g_hist);

figure;
subplot(2, 1, 1), imshow(green), title('Componente Verde');
subplot(2, 1, 2), imhist(green), axis auto, title('Histograma G');

% Transformación punto a punto de la variable intensidad 
% (Ver función GreenEttudes). 
[middle_green, trasference] = GreenEttudes(green, "linear");


% Se crea una nueva imagen "true color" cambiando la componente RGB verde 
% original por la nueva componente transformada.
mg_image = image;
mg_image(:, : ,2) = middle_green;


figure;
subplot(1, 1, 1), imshow(mg_image), title('Imagen con la componete verde modificada');

figure;
subplot(2, 1, 1), imshow(middle_green), title('Componente verde modificada');
subplot(2, 1, 2), imhist(middle_green), axis auto, title('Histograma G');

figure;
subplot(1, 1, 1), plot(trasference), title('mapa de transición'), grid on;
hold on, plot(d_range/3 * ones(1, 256), 1:256, "--r"), plot(2*d_range/3 * ones(1, 256), 1:256, "--r")
plot(d_range/4 * ones(1, 256), "--b"),plot(d_range/2 * ones(1, 256), "--b") 
plot(3*d_range/4 * ones(1, 256), "--b"), hold off


% Ecualización del histograma con la función "histeq" 
eq_green = histeq(middle_green);


imagen_eq = image;
imagen_eq(:, : ,2) = eq_green;
figure;
imshow(imagen_eq);



figure;
subplot(2, 1, 1), imshow(eq_green), title('Componente Verde tras EQ');
subplot(2, 1, 2), imhist(eq_green), axis auto, title('Histograma G');



%transfer = zeros(1706, 2560, 2);

%transfer(:,:,1) = middle_green;
%transfer(:,:,2) = eq_green;

%[x, y, z] = meshgrid(1:2560, 1:1706, 1:2);
%scatter3(x(:), y(:), z(:), 50, transfer(:), 'filled')

   


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Se convierte la imagen RGB a escala de grises.
gray = rgb2gray(image);



figure
subplot(1, 1, 1), imshow(gray), title('Imagen en escala de grises'); 


% Se obtiene el gradiente.
grad = imfilter(double(gray), fspecial("prewitt")');

figure
subplot(1, 1, 1), imshow(grad), title('Gradiente'); 

figure
subplot(1, 2, 1), mesh(double(grad)), colormap gray, zlabel("Nivel de intensidad"), title('Representación tridimensional');
subplot(1, 2, 2), mesh(double(gray)), colormap gray, zlabel("Nivel de intensidad"), title('Representación tridimensional');


%% Objetivo Creativo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc, clear all, close all

img = imread("G09.jpg");


% 1º: Cambiar del modelo RGB a YUV
YUV = rgb2ycbcr(img);

    % Extraer las componentes Y, U, V
Y = YUV(:, :, 1);
U = YUV(:, :, 2);
V = YUV(:, :, 1);

    % Mostrar la imagen original y las componentes Y, U, V
figure;
subplot(2, 2, 1), imshow(YUV), title('Imagen YUV');
subplot(2, 2, 2), imshow(Y), title('Y Component');
subplot(2, 2, 3), imshow(U), title('U Component');
subplot(2, 2, 4), imshow(V), title('V Component');

% 2º: Introducir un ruido a cada componente del modelo YUV
Y_gauss = imnoise(Y,'gaussian', 0, 0.03);
U_salt_pepper = imnoise(U,'salt & pepper', 0.07);
V_salt_pepper = imnoise(V,'salt & pepper', 0.09);
YUV_ruido = img;
YUV_ruido(:, :, 1) = Y_gauss;
YUV_ruido(:, :, 2) = U_salt_pepper;
YUV_ruido(:, :, 3) = V_salt_pepper;

figure;
subplot(3, 2, 1), imshow(Y_gauss), title('Y Component + Ruido Gaussiano');
subplot(3, 2, 2), imhist(Y_gauss), axis auto, title('Histograma Y + R.Gaussiano');
subplot(3, 2, 3), imshow(U_salt_pepper), title('U Component + Ruido Sal y Pimienta');
subplot(3, 2, 4), imhist(U_salt_pepper),  axis auto, title('Histograma Y + Ruido Sal y Pimienta');
subplot(3, 2, 5), imshow(V_salt_pepper), title('V Component + Ruido Sal y Pimienta');
subplot(3, 2, 6), imhist(V_salt_pepper), axis auto, title('Histograma V + R.Sal y Pimienta');

figure;
imshow(YUV_ruido), title('Imagen YUV + Ruido');

% 3º: Filtrado con filtros de suavizado para recuperar la imagen
    % 3.1º: F.P.B.Lineal + F.Mediana
    % Filtrado paso bajo lineal (suavizado)
mascara = (1/(8*8)*ones(8,8));
mascara_grande = (1/(30*30)*ones(30,30));

YUV_fgauss = imfilter(YUV_ruido, mascara,"symmetric", "same");

figure; 
subplot(2,1,1), imshow(YUV_fgauss), title('Filtro Gauss')
subplot(2,1,2), imhist(YUV_fgauss), axis auto, title('Histrograma F.Gauss')

YUV_fgauss_g = imfilter(YUV_ruido, mascara_grande,'symmetric','same');

figure;
subplot(2,1,1), imshow(YUV_fgauss_g), title('Filtro Gauss Máscara Grande');
subplot(2,1,2),imhist(YUV_fgauss_g), axis auto, title('Histograma F.Gauss Máscara Grande');

    % Filtrado paso bajo no lineal, filtro de mediana (muy útil para quitar el ruido de sal y pimienta)
Y_fgauss = YUV_fgauss(:, :, 1);
U_fgauss = YUV_fgauss(:, :, 2);
V_fgauss = YUV_fgauss(:, :, 3);

Y_fmediana_gauss = medfilt2(Y_fgauss, [15 15], 'symmetric');
U_fmediana_gauss = medfilt2(U_fgauss, [15 15], 'symmetric');
V_fmediana_gauss = medfilt2(V_fgauss, [15 15], 'symmetric');

YUV_fmediana_gauss = uint8(cat(3, Y_fmediana_gauss, U_fmediana_gauss, V_fmediana_gauss));

figure;
subplot(2,1,1), imshow(YUV_fmediana_gauss), title('Filtro Gaussiano + Filtro de mediana');
subplot(2,1,2), imhist(YUV_fmediana_gauss),  axis auto, title('Histograma F.P.B.lineal + F.Mediana');

    % 3.2º: F.Mediana + F.P.B.Lineal
    % Filtro de mediana
Y_fmediana =  medfilt2(Y_gauss, [15 15], 'symmetric');
U_fmediana =  medfilt2(U_salt_pepper, [15 15], 'symmetric');
V_fmediana =  medfilt2(V_salt_pepper, [15 15], 'symmetric');

YUV_fmediana = uint8(cat(3, Y_fmediana, U_fmediana, V_fmediana));

figure;
subplot(2,1,1), imshow(YUV_fmediana), title('F.Mediana');
subplot(2,1,2), imhist(YUV_fmediana),  axis auto;

    % Filtro paso bajo lineal
YUV_fgauss_mediana = imfilter(YUV_fmediana, mascara,"symmetric", "same");

figure;
subplot(2,1,1), imshow(YUV_fgauss_mediana), title('Filtro de mediana + Filtro Gaussiano');
subplot(2,1,2), imhist(YUV_fgauss_mediana), axis auto, title('Histrograma F.Mediana + F.P.B.lineal');

% 4º: Volver al modelo RGB
RGB_fmediana_gauss = ycbcr2rgb(YUV_fmediana_gauss);
RGB_fgauss_mediana = ycbcr2rgb(YUV_fgauss_mediana);

figure;
subplot(2, 2, 1), imshow(RGB_fmediana_gauss), title('Filtro Gaussiano + Filtro de mediana');
subplot(2, 2, 2), imhist(RGB_fmediana_gauss), axis auto, title('Histograma G + M');
subplot(2, 2, 3), imshow(RGB_fgauss_mediana), title('Filtro de mediana + Filtro Gaussiano');
subplot(2, 2, 4), imhist(RGB_fgauss_mediana), axis auto, title('Histograma M + G');

% Extra: Alternativas a la transformación lineal 
green = img(:, :, 2);
g_hist = imhist(green);
d_range = numel(g_hist);

[middle_green, trasference] = GreenEttudes(green, "linear");
[middle_green_ct, trasference_ct] = GreenEttudes(green, "const");
[middle_green_el, trasference_el] = GreenEttudes(green, "exp&log");


mg_image = img;
mg_image(:, : ,2) = middle_green;
mg_image_ct = img;
mg_image_ct(:, : ,2) = middle_green_ct;
mg_image_el = img;
mg_image_el(:, : ,2) = middle_green_el;

figure;
subplot(1, 1, 1), imshow(mg_image), title('Imagen con la componete verde modificada lineal');

figure;
subplot(1, 1, 1), plot(trasference), title('mapa de transición lineal'), grid on;
hold on, plot(d_range/3 * ones(1, 256), 1:256, "--r"), plot(2*d_range/3 * ones(1, 256), 1:256, "--r")
plot(d_range/4 * ones(1, 256), "--b"),plot(d_range/2 * ones(1, 256), "--b") 
plot(3*d_range/4 * ones(1, 256), "--b"), hold off

figure;
subplot(1, 1, 1), imshow(mg_image_ct), title('Imagen con la componete verde modificada constante');

figure;
subplot(1, 1, 1), plot(trasference_ct), title('mapa de transición constante'), grid on;
hold on, plot(d_range/3 * ones(1, 256), 1:256, "--r"), plot(2*d_range/3 * ones(1, 256), 1:256, "--r")
plot(d_range/4 * ones(1, 256), "--b"),plot(d_range/2 * ones(1, 256), "--b") 
plot(3*d_range/4 * ones(1, 256), "--b"), hold off

figure;
subplot(1, 1, 1), imshow(mg_image_el), axis auto, title('Imagen con la componete verde modificada log y exp');

figure;
subplot(1, 1, 1), plot(trasference_el), title('mapa de transición log y exp'), grid on;
hold on, plot(d_range/3 * ones(1, 256), 1:256, "--r"), plot(2*d_range/3 * ones(1, 256), 1:256, "--r")
plot(d_range/4 * ones(1, 256), "--b"),plot(d_range/2 * ones(1, 256), "--b") 
plot(3*d_range/4 * ones(1, 256), "--b"), hold off

figure
subplot(3, 1, 1),imhist(middle_green), axis auto, title('Histograma Transformación Lineal')
subplot(3, 1, 2),imhist(middle_green_ct), axis auto, title('Histograma Transformación Constante')
subplot(3, 1, 3),imhist(middle_green_el), axis auto, title('Histograma Transformación Logarítmica y Exponencial')


%% Funciones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Función encargada de realizar las transformaciones punto a punto. Tambien
% devuelve un vector con la transición de niveles, a fin de representarla.

% Recibe la matriz componente RGB verde y un string para seleccionar el
% tipo de transformación.


function [new_greens, tr_vector] = GreenEttudes(greens, type)

    % Se genera el histograma y el número de niveles.
    g_hist = imhist(greens);
    d_range = numel(g_hist(g_hist ~= 0));  
    
    % Por claridad en el código, se ha decidido recorrer un vector de una
    % dimensión mediante un bucle for. Se genera dicho vector. Tambien se
    % genera el vector transformación
    v_greens = double(reshape(greens, [1, numel(greens)]));
    tr_vector = zeros(d_range, 1);
    
    % La función genera tres tipos distintos de transformación.

    % Transformación constante:

    % Como el rango dinámico debe reducirse a la mitad, respetando el
    % tercio central original, se definen los tramos(0, d_range/4) y
    % (3*d_range/4, end). 
    % 
    % Las tres transformaciones alterarán el nivel de
    % todos los píxeles ubicados en estos tramos, asignando siempre un
    % nivel perteneciente a (d_range/4, d_range/3) o (2*d_range/3,
    % 3*d_range/4). Así, el tramo (d_range/3,2*d_range/3) nunca se ve
    % afectado por la transformación.


    if type == "const"
        % Esta transformación acumula todos los pixeles en los puntos
        % extremos del rango permitido. Se recorre el vector de pixeles, y
        % se genera el vector transformado asignando valores limites a los
        % pixeles con un bloque if else.


        for i =  1:numel(greens)
            if v_greens(i) < d_range/4
                v_greens(i) = d_range/4;
            elseif v_greens(i) > 3*d_range/4
                v_greens(i) = 3*d_range/4;
            else
                v_greens(i) = v_greens(i);
            end
        end

        % De forma análoga se recorren todos los niveles para obtener la
        % respuesta a la transformación en cada uno.

        for level = 1:256
            if level <  d_range/4
                tr_vector(level) = d_range/4;
            elseif level > 3*d_range/4
                tr_vector(level) = 3*d_range/4;
            else
                tr_vector(level) = level;
            end
        end

    elseif type == "linear"
        
        % En esta transformación en vez de asignar los valores extremos, se
        % asigna un valor según una ecuación lineal, tal que transforme de
        % forma uniforme los pixeles a lo largo de los rangos permitidos. 

        for i =  1:numel(v_greens)
            if v_greens(i) <= d_range/3
                v_greens(i) = (d_range/3 - d_range/4) / (d_range/3) * v_greens(i) + d_range/4;
            elseif greens(i) >= 2*d_range/3
                v_greens(i) = (3*d_range/4 - 2*d_range/3) / (d_range - 2*d_range/3) * v_greens(i) ... 
                + 2*d_range/3 - (3*d_range/4 - 2*d_range/3) / (d_range - 2*d_range/3) * 2*d_range/3 ;
            else
                v_greens(i) = v_greens(i);
            end
        end

        for level = 1:256
            if level < d_range/3
                tr_vector(level) = (d_range/3 - d_range/4) / (d_range/3) * level + d_range/4;
            elseif level > 2*d_range/3
                tr_vector(level) = (3*d_range/4 - 2*d_range/3) / (d_range - 2*d_range/3) * level ... 
                + 2*d_range/3 - (3*d_range/4 - 2*d_range/3) / (d_range - 2*d_range/3) * 2*d_range/3 ;
            else
                tr_vector(level) = level;
            end
        end   


        

    elseif type == "exp&log"

        % Esta transformación realiza la misma operación que la anterior,
        % pero en vez de con ecuaciones lineales, genera una curva
        % logarítmica en niveles bajos y una exponencial en niveles altos.

        gamma = 30;
        n = 10;

        for i =  1:numel(greens)
            if v_greens(i) <= d_range/3
                v_greens(i) = (d_range/3 - d_range/4) / logn(1 + d_range/3, n) * logn(1 + v_greens(i), n) + d_range/4;
            elseif v_greens(i) >= 2*d_range/3
                v_greens(i) = (3*d_range/4 - 2*d_range/3) / (d_range^gamma - (2*d_range/3)^gamma) * v_greens(i)^gamma ... 
                + 2*d_range/3 - (3*d_range/4 - 2*d_range/3) / (d_range^gamma - (2*d_range/3)^gamma) * (2*d_range/3)^gamma;
            else
                v_greens(i) = v_greens(i);
            end
        end

        for level = 1:256
            if level < d_range/3
                tr_vector(level) = (d_range/3 - d_range/4) / logn(1 + d_range/3, n) * logn(1 + level, n) + d_range/4;
            elseif level > 2*d_range/3
                tr_vector(level) = (3*d_range/4 - 2*d_range/3) / (d_range^gamma - (2*d_range/3)^gamma) * level^gamma ... 
                + 2*d_range/3 - (3*d_range/4 - 2*d_range/3) / (d_range^gamma - (2*d_range/3)^gamma) * (2*d_range/3)^gamma ;
            else
                tr_vector(level) = level;
            end
        end
    end
    
        
    new_greens = uint8(reshape(v_greens, size(greens)));
    
    % Implementación del logaritmo de base n. Durante el transcurso de la
    % tarea se consideró necesario, pero finalmente se decició utilizar
    % logaritmos con n = 10.

    function result = logn(x, n)
        result = log(x)/log(n);
    end

    

end
