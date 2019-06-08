# Trabalho Prático II: Ray Tracing

**Universidade Federal de Minas Gerais**   
Deiziane N. da Silva  
2015121980   
  
Prof. Erickson Nascimento  

## Introdução
Implementação básica de um Raytracer baseada no livro de Peter Shirley "_Ray Tracing in One Weekend_".

#### Como usar o programa:
O arquivo **raytracer.py** tem o main para este raytracer. É obrigatório informar o nome do arquivo de saída e opcional informar o tamanho
da imagem final. Caso o tamanho da imagem não seja informado, ela será renderizada com as dimensões 340x480.  

No _main_ você pode definir a resolução x e y da saída com **_width_** e **_height_** e o número de amostras por pixel com **_ns_**. 
Defina a imagem através da variável **_world_** para uma HittableList dos objetos que você deseja renderizar e configure a câmera 
como desejar. O valor **_depth_** define a profundidade máxima de recursão para o método de cor. É importante ressaltar que
após cerca de 5-10 saltos a maioria das imagens não muda muito (especialmente com muitas superfícies espelhadas). 


