# Computação Bioinspirada

A dieta nutricional é um fator importante para o desenvolvimento humano, tendo papel significativo na obtenção da melhor saúde. Entretanto, toda essa elaboração dietética apresenta problemas relacionados  à escolha de alimentos, quantidade de calorias ingeridas, tempo e principalmente otimização para obtenção de uma dieta balanceada. Assim, desenvolver uma dieta envolve cálculos e análises enormes de composições de alimentos e elementos nutritivos. Um forma de otimizar este problema é através da aplicação  de Algoritmos Genéticos (GA)  inspirados em modelos biológicos capazes de resolver problemas com rapidez e precisão. Assim, o presente trabalhos aplica GA ao problema da dieta para mono e multiobjetivos. Os resultados obtidos por simulação demonstram que a solução foi capaz de resolver o problema com rapidez e precisão, comprovando a eficiência do GA.


## Requisitos

Python 3.9 and the following libraries (check requirements.txt):

- autopep8==1.5.7
- cycler==0.10.0
- deap==1.3.1
- kiwisolver==1.3.1
- matplotlib==3.4.2
- numpy==1.19.5
- pandas==1.1.5
- Pillow==8.3.1
- pycodestyle==2.7.0
- pyparsing==2.4.7
- python-dateutil==2.8.2
- pytz==2021.1
- six==1.16.0
- toml==0.10.2

## Execução

Os parâmetros de calorias totais, período, e proporções dos macronutrientes devem ser definidos no início do arquivo diet.py. O programa pode ser executado apenas com:

```bash
python3 diet.py
```