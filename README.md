# Laboratório Quântico ⚛️

Uma aplicação web interativa desenvolvida com **Streamlit** para visualização e estudo de fenômenos fundamentais da mecânica quântica. O projeto oferece simulações físicas precisas com gráficos interativos utilizando **Plotly**.

## Funcionalidades

### 1. Poço de Potencial Infinito (1D)
Analise o comportamento ondulatório da matéria sob confinamento unidimensional.
* **Seleção de Partículas:** Escolha entre Elétrons, Prótons, Átomos de Hidrogênio, Hélio, Carbono ou até Buckyballs (C60).
* **Parâmetros Ajustáveis:** Controle a largura do poço (em nanômetros) e o número máximo de níveis de energia ($n$).
* **Visualização Tripla:**
  * Níveis de energia quantizados (eV).
  * Função de onda $\psi_n(x)$.
  * Densidade de probabilidade $|\psi_n(x)|^2$.

### 2. Orbitais Atômicos 3D e Átomos Multieletrônicos
Explore a parte angular da densidade de probabilidade dos orbitais atômicos e o efeito do núcleo.
* **Seleção de Elementos:** Veja como o aumento do número atômico ($Z$) contrai a nuvem eletrônica devido à atração eletrostática (Aproximação hidrogenoide).
* **Números Quânticos:** Ajuste os números quânticos azimutal ($l$) e magnético ($m_l$) para gerar orbitais s, p, d e f.
* **Gráficos 3D:** Renderização esférica interativa utilizando harmônicos esféricos complexos.

## 🛠️ Tecnologias e Bibliotecas

* **[Streamlit](https://streamlit.io/):** Criação da interface de usuário web.
* **[NumPy](https://numpy.org/):** Cálculos matriciais e físicos.
* **[SciPy](https://scipy.org/):** Computação de funções especiais (Harmônicos Esféricos).
* **[Plotly](https://plotly.com/python/):** Geração de gráficos 2D e 3D interativos.

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
