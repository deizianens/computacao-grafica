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
## Estrutura do Programa
### Ray tracing Básico
O programa básico foi implementado de acordo com o guia do livro do Peter Shirley. O programa implementa as classes: **Vec3**, **Ray**, **Hitable**, **HitableList**, **Sphere**, **Camera (Positional Camera)** e **Material (Lambertian, Metal, Dieletric)**. Com essas classes o programa consegue simular antialiasing, diferentes materiais e _depth of field_ (defocus blur).   
### Novas Superfícies 
Foram adicionados triângulos e carregamento de arquivos STL para renderizar objetos mais complicados.  
O formato de arquivo STL usa uma série de triângulos vinculados para recriar a geometria de superfície de um modelo sólido.  
### Reflexões imperfeitas
Muitos materiais são refletores imperfeitos, onde os reflexos são borrados em vários graus devido à aspereza da superfície que dispersa os raios das reflexões. Para conseguir esse efeito, as reflexões possuem um parâmetro **_fuzz_**. O material refletivo tem um coeficiente de reflexão, que define a porcentagem dos raios que será refletida. Podemos adicionar um fator fuzz, que aleatoriza a direção refletida usando uma pequena esfera e escolhendo um novo ponto final para o raio.

### Abertura e Distância Focal
Assim como numa câmera real, a cena desfoca em objetos que estejam longe do seu foco.   
Isso ocorre usando a randomização dos raios em função da sua distância da abertura.  
O processo é feito seguindo as instruções do livro.  
### Motion Blur
Para simular o efeito de movimento na imagem fazemos como acontece em uma câmera real:  
- Gerar raios em momentos aleatórios enquanto o obturador está aberto e interceptar o modelo naquele momento
adicionamos um parâmetro time, na classe Ray
- Adicionamos um intervalo de tempo no qual a câmera manda raios
- Adicionamos um objeto em movimento à cena, no caso, Moving Sphere onde seu centro move em um intervalo de tempo
## Resultados Obtidos
A primeira imagem gerada foi no tamanho 200x100, para proporcionar maior rapidez de execução. Na imagem há as 3 esferas principais (cada uma com um material diferente - Lambertiano, Dielétrico e Reflectivo) + 10 esferas pequenas geradas aleatóriamente.  
  
![[200x100](https://ibb.co/pQnjWWt)](https://i.ibb.co/xY3qmmp/1.jpg)

A segunda imagem gerada foi no tamanho 320x240, com apenas 50 esferas menores junto com as 3 principais. 

![[200x100](https://ibb.co/pQnjWWt)](https://i.ibb.co/7zxGd0J/3.jpg)
