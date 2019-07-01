# Trabalho Prático III: Animação

**Universidade Federal de Minas Gerais**   
Deiziane N. da Silva  
2015121980   
  
Prof. Erickson Nascimento  

## Introdução
Implementação em Python de um leitor de arquivos MD2 com renderização em OpenGL. O programa também implementa shaders e faz animação com shape interpolation.

#### Como usar o programa:
O arquivo **md2.py** tem o main para este programa. É obrigatório informar o nome do arquivo de entrada md2 e a animação desejada. Exemplo:  
> python md2.py STAND

## Estrutura do Programa
O programa básico foi implementado de acordo com o link disponibilizado como guia: http://tfc.duke.free.fr/old/models/md2.htm. 
Pela implementação ser em Python, algumas alterações foram necessárias (em C++ o David Henry faz muito uso de ponteiros, o que dificulta bastante a tradução de codigo)
Temos a classe principal MD2, que é responsável por ler o arquivo MD2 e carrega-lo. Também temos as classes de animação (Animate e AnimState), responsáveis por executar métodos relativos à animação do modelo.
As structs do código original foram implementadas aqui como classes Python (Frame, Triangle, TextCoord etc).

