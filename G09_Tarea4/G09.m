% Entrega 4
% G09
%     Diego Sota Rebollo
%     David Santa Cruz Del Moral
% 
% Objetivo Obligatorio 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all

% Carga la imagen RGB
I = imread("G09.jpg");

[num_filas, num_colum, ~] = size(I);

% Preprocesamiento de la imagen -> Cambio de dimensiones de la imagen
rgb = imresize(I, [num_filas / 8  num_colum / 8]);
 
figure;
imshow(rgb);

% Elección de la componente con la que se va a trabajar
% Separa los componentes RGB y los muestra
r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);

figure;
subplot(3,1,1), imshow(r), title('Componente Rojo');
subplot(3,1,2), imshow(g), title('Componente Verde');
subplot(3,1,3), imshow(b), title('Componente Azul');

% Convierte y muestra los componentes en el espacio de color Lab
lab = rgb2lab(rgb);
L = lab(:,:,1);
a = lab(:,:,2);
b = lab(:,:,3);

figure;
subplot(3,1,1), imshow(L, [0 100]), title('Componente L');
subplot(3,1,2), imshow(a, [0 100]), title('Componente a');
subplot(3,1,3), imshow(b, [0 100]), title('Componente b');

% Convierte y muestra los componentes en el espacio de color HSI
[h,s,i] = rgb2hsi(rgb);
hsi = cat(3, h, s, i);

figure;
subplot(3,1,1), imshow(h), title('Componente de Tono (H)');
subplot(3,1,2), imshow(s), title('Componente de Saturación (S)');
subplot(3,1,3), imshow(i), title('Componente de Intensidad (I)');


figure;
subplot(3,1,1), imshow(rgb);
subplot(3,1,2), imshow(hsi);
subplot(3,1,3), imshow(lab);

figure;
imshow(g); % Componente con la que nos vamos a quedar por la claridad 
% del objeto a segmentar (palo del paraguas)

% Preprocesado
%Top-Hat (Extracción de las partes más claras y más pequeñas que el EE)
se_Linea = strel('disk', 5);
g_topHat = imtophat(g, se_Linea);

figure;
imshow (g_topHat), title('G tras Top-Hat');

% Filtro alternado secuencial (aumenta el tamaño del EE). imagen más clara
% y reduce el ruido.
se_Linea = strel('line', 3, 60);
g_open = imopen(g_topHat, se_Linea);
g_fas1 = imclose(g_open, se_Linea);
figure;
subplot(2, 2, 1), imshow(g_fas1), title('g-fas1');

se_Linea = strel('line', 5, 60);
g_open2 = imopen(g_fas1, se_Linea);
g_fas2 = imclose(g_open2, se_Linea);
subplot(2, 2, 2), imshow(g_fas2), title('g-fas2');

se_Linea = strel('line', 7, 60);
g_open3 = imopen(g_fas2, se_Linea);
g_fas3 = imclose(g_open3, se_Linea);
subplot(2, 2, 3), imshow(g_fas3), title('g-fas3');

se_Linea = strel('line', 9, 60);
g_open4 = imopen(g_fas3, se_Linea);
g_fas4 = imclose(g_open4, se_Linea);
subplot(2, 2, 4), imshow(g_fas4), title('g-fas4');

figure;
% Nos quedamos con la tercera, ya que a partir de esta los 
% cambios apreciables son mínimos
imshow(g_fas3), title('G-Top-Hat tras el filtro alternado secuencial'); 

%%Umbralización 
% Histograma
figure;
imhist(g_fas3), title('Histograma');

% Tras observar el histograma el umbral nos interesa en 20.
umbral = graythresh(g_fas3);    % Umbral de imagen global usando el método de Otsu
I_bw = imbinarize(g_fas3, umbral);

figure;
imshow(I_bw), title('Imagen binaria');

% Apertura
% Para eliminar los elementos que no sean el palo del paraguas
% se procede a hacer una apertura de la imagen binaria.
seC = strel('square', 2);
seD = strel('diamond', 1);
seCir = strel('disk', 1);
seL = strel('line', 11, 60);

bw_apertura_c = imopen(I_bw, seC);
bw_apertura_d = imopen(I_bw, seD);
bw_apertura_cir = imopen(I_bw, seCir);
bw_apertura_l = imopen(I_bw, seL);

figure;
subplot(2, 2, 1), imshow(bw_apertura_c), title('EE = squere 2x2');
subplot(2, 2, 2),imshow(bw_apertura_d), title('EE = diamond 1');
subplot(2, 2, 3),imshow(bw_apertura_cir), title('EE = disk 1');
subplot(2, 2, 4),imshow(bw_apertura_l), title('EE = line 11 60');

% Nos quedaremos con la apertura hecha con el EE cuadrado ya que respeta un
% poco mejor los bordes del objeto de interés y elimina eficientemente todo 
% lo que no es este objeto (líneas pequeñas blancas en el centro de la imagen).

% Para hacer el elemento principal más ancho se hace una dilatación de la
% apertura pero con un elemento estructurante diferente (seL).
bw_apertura_c_dilation_l = imdilate(bw_apertura_c, seL);

figure;
imshow(bw_apertura_c_dilation_l), title('Apertura + Dilatación');


% Segmentación
% Segmentación binaria (busco grupos conexos de pixeles de primer plano)
[seg, num] = bwlabel(bw_apertura_c_dilation_l);
RGB_Segment = label2rgb(seg);

figure;
imshow(RGB_Segment)     % Imagen en falso color de la capa de etiquetas

% La etiqueta de interés es la etiqueta 1, la única que hay tras el
% preprocesado
[n_filas, n_cols] = size(bw_apertura_c_dilation_l);

b_max = [0, 0];
b_min = [99999, 99999];

% Bounding box
for ind_nfila=1:n_filas
    for ind_ncol=1:n_cols
        if bw_apertura_c_dilation_l(ind_nfila, ind_ncol) == 1
            if ind_nfila > b_max(1)
                b_max(1) = ind_nfila;
            end
            if ind_ncol > b_max(2)
                b_max(2) = ind_ncol;
            end
            if ind_nfila < b_min(1)
                b_min(1) = ind_nfila;
            end
            if ind_ncol < b_min(2)
                b_min(2) = ind_ncol;
            end
        
        end
    end
end

row_index = b_min(1):b_max(1);
col_index = b_min(2):b_max(2);

b_box = rgb(row_index, col_index, 1:3);

figure;
imshow(b_box), title('Imagen del tamaño de la bounding box que engloba al objeto segmentado');

% Objetivo Obligatorio 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Imagen RGB del objeto
I_palo = b_box;

%Procesado de imagen para eliminar intensidades bajas
% Para eliminar las intensidades bajas los operadores morfológicos que se
% pueden usar son la dilatación y el cierre. Se opta por usar el cierre ya
% que no modifica el tamaño del objeto.

se_Linea = strel('line', 11, 60);
se_Disco = strel('disk', 3);
se_Cuadrado = strel('square', 3);

I_cierre_Linea = imclose(I_palo, se_Linea);
I_cierre_Disco = imclose(I_palo, se_Disco);
I_cierre_Cuadrado = imclose(I_palo, se_Cuadrado);

figure;
subplot(1, 3, 1), imshow(I_cierre_Linea), title('I sin intensidades bajas EE = Línea');
subplot(1, 3, 2),imshow(I_cierre_Disco), title('I sin intensidades bajas EE = Disco');
subplot(1, 3, 3),imshow(I_cierre_Cuadrado), title('I sin intensidades bajas EE = Cuadrado');

% Observando los histogramas se puede observar que todos cumplen el
% objetivo de reducir intensidades bajas. Obviamente cada EE lo hace de una
% forma más o menos agresiva. Se puede concluir que al eliminar las
% intensidades más bajas se aumentan otras intensidades más altas.
figure;
subplot(2, 2, 1), imhist(I_palo), axis auto, title('Histograma imagen del objeto segmentado');
subplot(2, 2, 2), imhist(I_cierre_Linea), axis auto, title('Histograma imagen del objeto tras el cierre EE = L');
subplot(2, 2, 3), imhist(I_cierre_Disco), axis auto, title('Histograma imagen del objeto tras el cierre EE = D');
subplot(2, 2, 4), imhist(I_cierre_Cuadrado), axis auto, title('Histograma imagen del objeto tras el cierre EE = C');


%% Objetivo Creativo %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear all, close all

% Carga la imagen RGB
rgb = imread("G09.jpg");

[num_filas, num_colum, ~] = size(rgb);

figure;
imshow(rgb);

r = rgb(:,:,1);
b = rgb(:,:,3);

% La componente verde solo aporta información útil del poste.
% se utiliza la bounding blox del objetivo obligatorio para filtrarlo.

b_box_row = (113*8):(214*8 - 16);
b_box_col = (86*8):(157*8);
del_box_row = (113*8):(125*8);
del_box_col = (86*8):(140*8);

aux = uint8(zeros(num_filas, num_colum));
aux(b_box_row, b_box_col) = 1;
aux(del_box_row, del_box_col) = 0;
g = aux .* rgb(:,:,2);

figure;
subplot(1,3,1), imshow(r);
subplot(1,3,2), imshow(b);
subplot(1,3,3), imshow(g);

% Top-Hat y Bot-Hats

SE = strel('disk', 11);
SE2 = strel('disk', 23);
r_botHat = imbothat(r, SE);
b_botHat = imbothat(b, SE);
g_topHat = imtophat(g, SE2);

% Suma ponderada de componentes transformadas.

total = r_botHat + 2*b_botHat + g_topHat;

figure;
imshow (r_botHat), title('R tras Bot-Hat');

figure;
imshow (b_botHat), title('B tras Bot-Hat');

figure;
imshow (g_topHat), title('G tras Top-Hat');

figure;
imshow (total), title('r total');

% Binarización

I_bw = imbinarize(total, 0.3);

figure;
imshow(I_bw)

% Selección de color según capa de etiqueta.

creative_image = rgb;

grey_image = rgb2gray(creative_image);

creative_image(:,:,1) = grey_image;
creative_image(:,:,2) = grey_image;
creative_image(:,:,3) = grey_image;

for i = 1:num_filas
    for j = 1:num_colum
        if I_bw(i, j) == 1
            creative_image(i,j,:) = rgb(i,j,:);
        end
    end
end


% Resultado Final

figure;
imshow(creative_image)


% Metodo 2 

% Gradiente Morfológico

SE = strel('disk',7);

gray_eroded = imerode(grey_image, SE);
gray_dilated = imdilate(grey_image, SE);
figure;
imshow(gray_eroded)
figure;
imshow(gray_dilated)

morph_grad = gray_dilated - gray_eroded;

figure;
imshow(morph_grad)

% Umbralización

umbral = graythresh(morph_grad);    % Umbral de imagen global usando el método de Otsu
I_bw2 = imbinarize(morph_grad, umbral);

figure;
imshow(I_bw2)

% Selección de color según capa de etiqueta.

creative_image = rgb;

grey_image = rgb2gray(creative_image);

creative_image(:,:,1) = grey_image;
creative_image(:,:,2) = grey_image;
creative_image(:,:,3) = grey_image;

for i = 1:num_filas
    for j = 1:num_colum
        if I_bw2(i, j) == 1
            creative_image(i,j,:) = rgb(i,j,:);
        end
    end
end

% Resultado Final

figure;
imshow(creative_image)








