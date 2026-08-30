# Laboratório Quântico ⚛️

Uma aplicação web interativa desenvolvida com Streamlit para visualização e estudo de fenômenos fundamentais da mecânica quântica. O projeto oferece simulações físicas e visualizações interativas utilizando Plotly, permitindo explorar conceitos como quantização de energia, tunelamento quântico e orbitais atômicos.

## Funcionalidades

### 1. Poço de Potencial Infinito (1D)

Analise o comportamento ondulatório da matéria sob confinamento unidimensional.

* **Seleção de Partículas:** Escolha entre Elétrons, Prótons, Átomos de Hidrogênio, Hélio, Carbono ou até Buckyballs (C60).
* **Parâmetros Ajustáveis:** Controle a largura do poço (em nanômetros) e o número máximo de níveis de energia ($n$).
* **Visualização Tripla:**

  * Níveis de energia quantizados (eV).
  * Função de onda $\psi_n(x)$.
  * Densidade de probabilidade $|\psi_n(x)|^2$.

### 2. Tunelamento Quântico

Explore o comportamento de uma partícula quântica ao incidir sobre uma barreira de potencial retangular.

* **Parâmetros Ajustáveis:** Controle a largura da barreira ($a$), sua altura ($V_0$) e a energia do elétron ($E$).
* **Coeficiente de Transmissão:** Calcule numericamente a probabilidade de o elétron atravessar a barreira.
* **Coeficiente de Reflexão:** Visualize a parcela da onda que é refletida pela barreira.
* **Dois Regimes Quânticos:** Explore tanto o regime de tunelamento ($E < V_0$), quanto o caso em que a energia da partícula supera a barreira ($E > V_0$).
* **Visualização da Função de Onda:** Observe a parte real da função de onda $\mathrm{Re}[\psi(x)]$ nas regiões incidente, da barreira e transmitida.
* **Densidade de Probabilidade:** Visualize $|\psi(x)|^2$ e observe o decaimento da função de onda no interior da barreira durante o tunelamento.
* **Potencial Interativo:** O potencial $V(x)$ e a energia da partícula são representados diretamente no gráfico para facilitar a interpretação física do fenômeno.

### 3. Orbitais Atômicos 3D

Explore a densidade de probabilidade tridimensional dos elétrons no átomo de Hidrogênio a partir da solução da Equação de Schrödinger.

* **Números Quânticos:** Ajuste os números quânticos principal ($n$), azimutal ($l$) e magnético ($m_l$), respeitando as condições quânticas permitidas.
* **Diferentes Orbitais:** Explore orbitais dos tipos s, p, d, f, g e h conforme os valores permitidos de $n$ e $l$.
* **Harmônicos Esféricos:** A forma angular dos orbitais é calculada utilizando os harmônicos esféricos complexos.
* **Nós Radiais:** Identifique a quantidade de nós radiais através da relação $n-l-1$.
* **Nós Angulares:** Observe a influência de $l$ sobre a estrutura espacial do orbital.
* **Dependência de $m_l$:** Explore como o número quântico magnético modifica a distribuição angular do orbital.
* **Visualização 3D Interativa:** Renderização tridimensional dos orbitais utilizando Plotly, com escala de cores representando a densidade de probabilidade $|\psi|^2$.
* **Parte Radial:** A função radial $R_{nl}(r)$ é calculada a partir dos polinômios associados de Laguerre, incorporando a dependência dos estados quânticos na representação.

## 🛠️ Tecnologias e Bibliotecas

* **Streamlit:** Criação da interface de usuário web.
* **NumPy:** Cálculos numéricos, matriciais e manipulação dos dados das simulações.
* **SciPy:** Cálculo de funções especiais, incluindo harmônicos esféricos e polinômios associados de Laguerre.
* **Plotly:** Geração de gráficos 2D e visualizações 3D interativas.

## Instalação e Execução

1. Clone este repositório ou baixe o código-fonte.

2. Instale as dependências necessárias utilizando o `pip`:

   ```bash
   pip install streamlit numpy scipy plotly
   ```

3. Execute a aplicação:

   ```bash
   streamlit run app.py
   ```

4. Utilize o menu lateral para selecionar entre as três simulações disponíveis e altere os parâmetros para explorar os diferentes fenômenos quânticos.


## Vídeo

O link abaixo contém um pequeno vídeo o qual explica a ideia geral da aplicação.

https://youtu.be/RpsaLS-utKU
