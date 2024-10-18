% Entrega 3
% G09
%     Diego Sota Rebollo
%     David Santa Cruz Del Moral
% 
%% %%%%%%%%%%%%%%%%%%% OBJETIVO OBLIGATORIO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    clc, clear all, close all


% Carga la imagen RGB

    I = imread("G09.jpg");
    [num_filas, num_colum] = size(I);


% Preprocesamiento de la imagen -> Cambio de dimensiones de la imagen

    rgb = imresize(I, [num_filas/8, num_colum/(8*3)]);
    mascara = (1/(7*7)*ones(7,7));

        figure; % Figura 1
        imshow(rgb);
        [rows, cols] = size(rgb(:,:,1));


% Separa los componentes RGB y los muestra

    fr = medfilt2(rgb(:,:,1), [12, 12]);
    fg = medfilt2(rgb(:,:,2), [12, 12]);
    fb = medfilt2(rgb(:,:,3), [12, 12]);

    fRGB = cat(3, fr, fg, fb);

        figure; % Figura 2
        imshow(fRGB);


% Convierte y muestra los componentes en el espacio de color Lab

    lab = rgb2lab(fRGB);
    L = lab(:,:,1);
    a = lab(:,:,2);
    b = lab(:,:,3);
    
        figure; % Figura 3
        subplot(3,1,1), imshow(L, [0, 100]), title('Componente L');
        subplot(3,1,2), imshow(a, [0, 100]), title('Componente a');
        subplot(3,1,3), imshow(b, [0, 100]), title('Componente b');


% Convierte y muestra los componentes en el espacio de color HSI
    [h,s,i] = rgb2hsi(fRGB);
    hsi = cat(3, h, s, i);
    
        figure; % Figura 4
        subplot(3,1,1), imshow(h), title('Componente de Tono (H)');
        subplot(3,1,2), imshow(s), title('Componente de Saturación (S)');
        subplot(3,1,3), imshow(i), title('Componente de Intensidad (I)');
        
        figure; % Figura 5
        subplot(3,1,1), imshow(fRGB);
        subplot(3,1,2), imshow(hsi);
        subplot(3,1,3), imshow(lab);


% Características y scatter plot
    ctrsAB = [reshape(a, 1, []); reshape(b, 1, [])]';
    ctrsHS = [reshape(h, 1, []); reshape(s, 1, [])]';
    
        figure, % Figura 6
        subplot(2,1,1)
        plot(ctrsHS(:,1), ctrsHS(:,2),'.')
        title('Scatter plot con observaciones H-S')
        xlabel('h'), ylabel('s')
        axis equal
        
        subplot(2,1,2) % Figura 7
        plot(ctrsAB(:,1), ctrsAB(:,2),'.')
        title('Scatter plot con observaciones A-B')
        xlabel('a'), ylabel('b')
        axis equal


% Estandarización de características
    ndim = size(ctrsAB,2);
    ctrsAB_norm = ctrsAB;
    V_means = zeros(1,ndim);  % variable para almacenar la media de cada caracteristica
    V_std = zeros(1,ndim);  % variable para almacenar la desviacion tipica de cada caracteristica
    
    for ind_dim=1:ndim
         datos = ctrsAB(:,ind_dim); % seleccionamos todos los valores de una caracteristica
         V_means(ind_dim) = mean(datos);
         V_std(ind_dim) = std(datos);
         datos_norm = (datos-V_means(ind_dim))/V_std(ind_dim);
         ctrsAB_norm(:, ind_dim)=datos_norm;
    end

        figure; % Figura 8
        plot(ctrsAB_norm(:,1), ctrsAB_norm(:,2),'.')
        title('Scatter plot con observaciones estandarizadas A-B')
        xlabel('a'), ylabel('b')
        axis equal


% Aplicación del algoritmo k-medias 

    % Inicialización de los centroides de forma aleatoria
    ngrupos = 8;
    [cluster_idx_norm, cluster_center_norm] = kmeans(ctrsAB_norm,ngrupos,'distance','sqEuclidean','Replicates',10);
    
    cluster_center_unnorm = zeros(8, 2);
    cluster_center_unnorm(:, 1) = cluster_center_norm(:, 1) .* V_std(1) + V_means(1);
    cluster_center_unnorm(:, 2) = cluster_center_norm(:, 2) .* V_std(2) + V_means(2);
    
        figure; % Figura 9
        plot(ctrsAB_norm(:,1), ctrsAB_norm(:,2),'.')
        title('Scatter plot, centroides')
        xlabel('a'), ylabel('b')
        hold on
        plot(cluster_center_norm(:,1), cluster_center_norm(:,2),'sr','MarkerSize',20, 'MarkerEdgeColor','r');

  




    % Obtención de los colores de cada región. 
    % Representación de la nueva imagen.

    new_image = zeros(rows,cols,3);
    [new_image(:,:,1), lumas] = mean_luma(cluster_idx_norm, L, rows, cols);
    [new_image(:,:,2), new_image(:,:,3)] = centroid_chromas(cluster_idx_norm, cluster_center_unnorm, rows, cols);

    new_rgb_image = lab2rgb(new_image);
    palet = cat(2, lumas(:,1)./lumas(:, 2) , cluster_center_unnorm);
    selected_palet = lab2rgb(palet);

        figure; imshow(new_rgb_image); % Figura 10
        
    % Scatter plot con colores consistentes con la imagen previa.
        figure; % Figura 11
        title('Segmentación 8-medias')
        xlabel('a'), ylabel('b')
        hold on
        i1_norm = find(cluster_idx_norm==1);
        i2_norm = find(cluster_idx_norm==2);
        i3_norm = find(cluster_idx_norm==3);
        i4_norm = find(cluster_idx_norm==4);
        i5_norm = find(cluster_idx_norm==5);
        i6_norm = find(cluster_idx_norm==6);
        i7_norm = find(cluster_idx_norm==7);
        i8_norm = find(cluster_idx_norm==8);
        
        plot(ctrsAB_norm(i1_norm,1), ctrsAB_norm(i1_norm,2), '.', Color=selected_palet(1,:));
        plot(ctrsAB_norm(i2_norm,1), ctrsAB_norm(i2_norm,2),'.', Color=selected_palet(2,:));
        plot(ctrsAB_norm(i3_norm,1), ctrsAB_norm(i3_norm,2),'.', Color=selected_palet(3,:));
        plot(ctrsAB_norm(i4_norm,1), ctrsAB_norm(i4_norm,2),'.', Color=selected_palet(4,:));
        plot(ctrsAB_norm(i5_norm,1), ctrsAB_norm(i5_norm,2),'.', Color=selected_palet(5,:));
        plot(ctrsAB_norm(i6_norm,1), ctrsAB_norm(i6_norm,2),'.', Color=selected_palet(6,:));
        plot(ctrsAB_norm(i7_norm,1), ctrsAB_norm(i7_norm,2),'.', Color=selected_palet(7,:));
        plot(ctrsAB_norm(i8_norm,1), ctrsAB_norm(i8_norm,2),'.', Color=selected_palet(8,:));
        plot(cluster_center_norm(:,1), cluster_center_norm(:,2),'sr','MarkerSize',20, 'MarkerEdgeColor','r');
        axis equal 
    

    % Inicialización manual de los centroides
    centroides =  [-1.61, 1.17;
                -0.28, 1.04; 
                1.25, 0.95;  
                1.41, -0.82;  
                -0.86, 0.93; 
                0.97, -1.73; 
                0.04, -0.91 
                -0.50, -1.16 
                ];

    ngrupos = 8;
    [cluster_idx_norm2, cluster_center_norm2] = kmeans(ctrsAB_norm,ngrupos,'distance','sqEuclidean','Replicates',1, 'Start', centroides);
    
    % Idéntico al caso anterior
    cluster_center_unnorm2 = zeros(8, 2);
    cluster_center_unnorm2(:, 1) = cluster_center_norm2(:, 1) .* V_std(1) + V_means(1);
    cluster_center_unnorm2(:, 2) = cluster_center_norm2(:, 2) .* V_std(2) + V_means(2);

    new_image2 = zeros(rows,cols,3);
    [new_image2(:,:,1), lumas2] = mean_luma(cluster_idx_norm2, L, rows, cols);
    [new_image2(:,:,2), new_image2(:,:,3)] = centroid_chromas(cluster_idx_norm2, cluster_center_unnorm2, rows, cols);
    

    new_rgb_image2 = lab2rgb(new_image2);
    palet2 = cat(2, lumas2(:,1)./lumas2(:, 2) , cluster_center_unnorm2);
    selected_palet2 = lab2rgb(palet2);

        figure; imshow(new_rgb_image2); % Figura 12
    
        figure; % Figura 13
        plot(ctrsAB_norm(:,1), ctrsAB_norm(:,2),'.')
        title('Scatter plot, centroides')
        xlabel('a'), ylabel('b')
        hold on
        plot(cluster_center_norm2(:,1), cluster_center_norm2(:,2),'sr','MarkerSize',20, 'MarkerEdgeColor','r');
        
        
        figure; % Figura 14
        title('Segmentación 8-medias')
        xlabel('a'), ylabel('b')
        hold on
        i1_norm2 = find(cluster_idx_norm2==1);
        i2_norm2 = find(cluster_idx_norm2==2);
        i3_norm2 = find(cluster_idx_norm2==3);
        i4_norm2 = find(cluster_idx_norm2==4);
        i5_norm2 = find(cluster_idx_norm2==5);
        i6_norm2 = find(cluster_idx_norm2==6);
        i7_norm2 = find(cluster_idx_norm2==7);
        i8_norm2 = find(cluster_idx_norm2==8);
        
        plot(ctrsAB_norm(i1_norm2,1), ctrsAB_norm(i1_norm2,2), '.', Color=selected_palet2(1,:));
        plot(ctrsAB_norm(i2_norm2,1), ctrsAB_norm(i2_norm2,2),'.', Color=selected_palet2(2,:));
        plot(ctrsAB_norm(i3_norm2,1), ctrsAB_norm(i3_norm2,2),'.', Color=selected_palet2(3,:));
        plot(ctrsAB_norm(i4_norm2,1), ctrsAB_norm(i4_norm2,2),'.', Color=selected_palet2(4,:));
        plot(ctrsAB_norm(i5_norm2,1), ctrsAB_norm(i5_norm2,2),'.', Color=selected_palet2(5,:));
        plot(ctrsAB_norm(i6_norm2,1), ctrsAB_norm(i6_norm2,2),'.', Color=selected_palet2(6,:));
        plot(ctrsAB_norm(i7_norm2,1), ctrsAB_norm(i7_norm2,2),'.', Color=selected_palet2(7,:));
        plot(ctrsAB_norm(i8_norm2,1), ctrsAB_norm(i8_norm2,2),'.', Color=selected_palet2(8,:));
        plot(cluster_center_norm2(:,1), cluster_center_norm2(:,2),'sr','MarkerSize',20, 'MarkerEdgeColor','r');
        axis equal
    
        
        % Representación de la componente de luminancia, y de su
        % histograma. 

        figure; % Figura 15
        subplot(2,1,1);imshow(new_image2(:,:,1),[0 100])
        subplot(2,1,2);histogram(new_image2(:,:,1), 100);
        
    

% Segmentación a partir de componente L     
    borders = front_markers(L, lumas2, rows, cols);

    % Se genera una imagen para cada objeto a segmentar.
    figure; imshow(red_borders(rgb, borders, rows, cols, 1)) % Figura 16
    figure; imshow(red_borders(rgb, borders, rows, cols, 2)) % Figura 17
    figure; imshow(red_borders(rgb, borders, rows, cols, 3)) % Figura 18
    figure; imshow(red_borders(rgb, borders, rows, cols, 4)) % Figura 19
    figure; imshow(red_borders(rgb, borders, rows, cols, 5)) % Figura 20
    figure; imshow(red_borders(rgb, borders, rows, cols, 6)) % Figura 21
    figure; imshow(red_borders(rgb, borders, rows, cols, 7)) % Figura 22
    figure; imshow(red_borders(rgb, borders, rows, cols, 8)) % Figura 23



%% %%%%%%%%%%%%%%%%%%%%%%% OBJETIVO CREATIVO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    clc, clear all, close all


% Cargar la imagen y redimensionarla

    I = imread("G09.jpg");
    [num_filas, num_colum] = size(I);
    I_peq = imresize(I, [num_filas/8  num_colum/(8*3)]);
    
        figure; % Figura 1
        imshow(I_peq), title('Imagen redimensionada');

                    % Función para extraer coordenadas pulsano la imagen
                    % [x, y] = ginput(3);  
                    % 
                    % % Mostrar los puntos seleccionados
                    % hold on;
                    % plot(x, y, 'r*');


% Elección de pequeñas muestras (triángulos) que caracterizan los colores presentes en la imagen.

    val = zeros(3, 2, 8); % Inicialización de 'val'. 3*2 (vertices de triángulos) y 9 muestras.


% Coordenadas de las muestras de los nueve colores presentes en la imagen.

    val(:,:,1) = [                          % Rojo
    
       67.6526162790698     83.7049418604651
       73.8735465116279     105.789244186047
       104.978197674419     89.3037790697674
    
    ];
    
    val(:,:,2) = [                          % Rosa
    
       2.64389534883722     165.821220930233
       78.5392441860465     212.167151162791
       135.149709302326     121.341569767442
    
    ];
    
    val(:,:,3) = [                          % Morado
    
       135.162400000000     204.399200000000
       189.261600000000     204.056800000000
       145.776800000000     177.692000000000
    
    ];
    
       
    val(:,:,4) = [                          % Azul Oscuro
    
       177.140988372093     119.786337209302
       203.268895348837     211.234011627907
       304.981104651163     206.568313953488
    
    ];
    
    
    val(:,:,5) = [                          % Cian
    
       182.117732558140     104.234011627907
       316.800872093023     76.5508720930233
       318.045058139535     187.594476744186
    
    ];
    
    
    val(:,:,6) = [                          % Verde
    
       175.896802325581     93.0363372093023
       223.175872093023     5.94331395348837
       316.489825581395     52.9113372093023
       
    ];
    
    
    val(:,:,7) = [                          % Amarillo
    
       135.771802325581     1.58866279069767
       209.800872093023     3.14389534883719
       166.254360465116     85.5712209302326
    
    ];
    
    val(:,:,8) = [                          % Naranja
    
       148.213662790698     85.2601744186047
       127.684593023256     4.38808139534882
       13.2194767441861     27.4055232558139
    
    ];
    
    val(:,:,9) = [                          % Negro
    
       153.190406976744     110.454941860465
       156.611918604651     107.344476744186
       159.100290697674     111.077034883721   
    
    ];


% Cargar 'val' en 'region_coordinates'
    sample_coordinates = val;


% Creación de regiones de muestra.

    % Se crea una matriz lógica para cada color en la imagen. 
    % Para cada color, se crea una región de interés basada en las coordenadas de las muestras.
    num_colors = 9;
    sample_regions = false([size(I_peq,1) size(I_peq,2) num_colors]);

    for count = 1:num_colors
      sample_regions(:,:,count) = roipoly(I_peq,sample_coordinates(:,1,count), ...
          sample_coordinates(:,2,count));
    end

        figure;  % Figura 2    
        imshow(sample_regions(:,:,3)), title("Región de muestra morado");


% Conversión de la imagen al espcaio LAB y extración de las componentes a y b.

    I_lab = rgb2lab(I_peq);
    a = I_lab(:,:,2);
    b = I_lab(:,:,3);

        figure;  % Figura 3
        imshow(I_lab), title('Imagen espacio LAB'); 


% Extracción de marcadores de color (media de a y b) dentro de cada región.
    color_markers = zeros([num_colors, 2]);
    
    for count = 1:num_colors
      color_markers(count,1) = mean2(a(sample_regions(:,:,count)));
      color_markers(count,2) = mean2(b(sample_regions(:,:,count)));
    end


% Clasificación de cada píxel usando la regla del vecino más próximo.
    color_labels = 0:num_colors-1;   % Etiquetas de color.
    
    a = double(a);
    b = double(b);
    distance = zeros([size(a),num_colors]);
    
    for count = 1:num_colors
      distance(:,:,count) = ( (a - color_markers(count,1)).^2 + (b - color_markers(count,2)).^2 ).^0.5;
    end

    [~,label] = min(distance,[],3);
    label = color_labels(label);
    clear distance;


% Mostrar los resultados de la clasificación.
    labels = repmat(label,[1 1 3]);
    segmented_img = zeros([size(I_peq),num_colors],"uint8");
    
    for count = 1:num_colors
      color = I_peq;
      color(labels ~= color_labels(count)) = 0;
      segmented_img(:,:,:,count) = color;
    end 
        
        figure; % Figura 4
        montage({segmented_img(:,:,:,8),segmented_img(:,:,:,7), segmented_img(:,:,:,6) ...
            segmented_img(:,:,:,1), segmented_img(:,:,:,9),segmented_img(:,:,:,5) ... 
            segmented_img(:,:,:,2), segmented_img(:,:,:,3), segmented_img(:,:,:,4) ...
        });

        title("Separación de los colores del paraguas: Rosa, Morado, Azul, Cian, Verde, Amarillo, Naranja, Negro y Rojo")


% Valores de A y B etiquetados.
    purple = "#774998";
    orange = "#FF8800";
    plot_labels = ["k", "r", "g", purple, "m", "y", "b", orange, "c"];
    
        figure;
        for count = 1:num_colors
            plot_label = plot_labels(count);
            plot(a(label==count-1), b(label==count-1),".", MarkerEdgeColor=plot_label, MarkerFaceColor=plot_label);
            hold on
        end
        title("Scatterplot Espacio A B"), xlabel("a"), ylabel("b");



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function Img = red_borders(RGB, borders, r, c, tag)

            Img = RGB;
            for j = 1:r
                for i = 1:c
                    if borders(j, i) == tag
                        Img(j, i, :) = [255, 0, 0];
                    end
                end
            end
        end
        

        function map = front_markers(L, umbrals , r, c)

            map = zeros(r, c, 1);
            markers = zeros(r, c);

            for j = 1:r
                for i = 1:c
                    [~, idx] = min(abs((umbrals(:, 1)./ umbrals(:, 2))- L(j, i)));
                    markers(j, i) = idx;
                end
            end
            
            for j = 2:r-1
                for i = 2:c-1
                    for val = 1:8
                        if markers(j, i) == val

                            is_border = ((markers(j-1, i-1) == val) | (markers(j-1, i) == val) | (markers(j-1, i+1) == val) | ...
                                (markers(j, i-1) == val) | markers(j, i) == val | (markers(j, i+1) == val) | ...
                                (markers(j+1, i-1) == val) | (markers(j+1, i) == val) | (markers(j+1, i+1) == val)) & ... 
                                ~((markers(j-1, i-1) == val) & (markers(j-1, i) == val) & (markers(j-1, i+1) == val) & ...
                                (markers(j, i-1) == val) & markers(j, i) == val & (markers(j, i+1) == val) & ...
                                (markers(j+1, i-1) == val) & (markers(j+1, i) == val) & (markers(j+1, i+1) == val));

                            if is_border
                                map(j, i, 1) = val;
                            end

                        end
                    end
                end
            end
        end

    
    function [a, b] = centroid_chromas(markers, centroids, r, c)

        map = reshape(markers, r, c);
        a = zeros(r, c);
        b = zeros(r, c);
        for j = 1:r
            for i = 1:c
                a(j, i) = centroids(map(j, i), 1);
                b(j, i) = centroids(map(j, i), 2);
            end
        end
    end
    
    
    function [luma, aux] = mean_luma(markers, L, r, c)
    
        map = reshape(markers, r, c);
        luma = zeros(r, c, 1);
        aux = zeros(8, 2);
    
        for j = 1:r
            for i = 1:c
                aux(map(j, i), 1) = aux(map(j, i), 1) + L(j, i);
                aux(map(j, i), 2) = aux(map(j, i), 2) + 1;
            end
        end
    
        means = aux(:, 1) ./ aux(:, 2);
    
        for j = 1:r
            for i = 1:c
                luma(j,i,1) = means(map(j, i));
            end
        end
    end
    
    
    

