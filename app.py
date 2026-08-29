import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.special as sp

# --- Configuração da Página ---
st.set_page_config(
    layout="wide",
    page_title="Laboratório Quântico",
    page_icon="⚛️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #00e5ff; font-weight: 300; }
    </style>
""", unsafe_allow_html=True)

# --- Constantes Físicas ---
hbar = 1.054571817e-34  # J*s
eV_to_J = 1.60218e-19
amu_to_kg = 1.66053906660e-27

particles = {
    "Elétron": 9.109e-31,
    "Próton": 1.672e-27,
    "Átomo de Hidrogênio (1u)": 1 * amu_to_kg,
    "Átomo de Hélio (4u)": 4 * amu_to_kg,
    "Átomo de Carbono (12u)": 12 * amu_to_kg,
    "Buckyball C60 (720u)": 720 * amu_to_kg
}


def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Converte uma cor hexadecimal (#rrggbb) para string rgba() com transparência."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Calcula Y_l^m(theta, phi) de forma compatível com scipy novo e antigo."""
    if hasattr(sp, "sph_harm_y"):
        return sp.sph_harm_y(l, m, theta, phi)
    return sp.sph_harm(m, l, phi, theta)


def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """
    Parte radial R_nl(r) do átomo de Hidrogênio (Z=1), com r em raios de Bohr (a0=1).
    Já normalizada: integral de |R_nl(r)|^2 * r^2 dr = 1.
    """
    rho = 2 * r / n
    norm = np.sqrt((2 / n) ** 3 * sp.factorial(n - l - 1) / (2 * n * sp.factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho ** l * sp.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


# ==========================================
# MENU DE NAVEGAÇÃO
# ==========================================
st.sidebar.image(
    "https://www3.unicentro.br/petfisica/wp-content/uploads/sites/54/2025/05/Dia-da-Mecanica-quantica.jpg",
    width=100
)
st.sidebar.markdown("## Navegação")
menu = st.sidebar.radio("Escolha a simulação:", [
    "Poço de Potencial Infinito",
    "Tunelamento",
    "Orbitais Atômicos 3D"
])
st.sidebar.markdown("---")

# ==========================================
# SIMULAÇÃO 1: POÇO DE POTENCIAL INFINITO
# ==========================================
if menu == "Poço de Potencial Infinito":
    with st.sidebar:
        st.markdown("## Parâmetros do Poço")
        selected_particle = st.selectbox("Elemento/Partícula:", list(particles.keys()))
        mass_kg = particles[selected_particle]

        L_nm = st.slider("Largura do Poço (nm):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        L_m = L_nm * 1e-9

        n_max = st.number_input("Número máximo de níveis (n):", min_value=1, max_value=10, value=4)

    x = np.linspace(0, L_nm, 500)
    E1_J = (np.pi ** 2 * hbar ** 2) / (2 * mass_kg * L_m ** 2)
    E1_eV = E1_J / eV_to_J

    st.title("Poço de Potencial Infinito")
    st.markdown("Analise o comportamento ondulatório da matéria sob confinamento absoluto (1D).")

    col1, col2, col3 = st.columns(3)
    col1.metric("Massa da Partícula", f"{mass_kg:.2e} kg")
    col2.metric("Largura do Poço (L)", f"{L_nm:.1f} nm")
    col3.metric("Energia Fundamental (E₁)", f"{E1_eV:.2e} eV")

    st.markdown("---")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Níveis de Energia", "Função de Onda ψ<sub>n</sub>(x)", "Probabilidade |ψ<sub>n</sub>(x)|²"),
        horizontal_spacing=0.05
    )

    colors = [
        '#00e5ff', '#ff00ea', '#76ff03', '#ffea00', '#ff3d00',
        '#bd00ff', '#1de9b6', '#ff4081', '#40c4ff', '#eeff41'
    ]
    max_energy = E1_eV * (n_max ** 2)

    for n in range(1, n_max + 1):
        En_eV = E1_eV * (n ** 2)
        psi = np.sqrt(2 / L_nm) * np.sin(n * np.pi * x / L_nm)
        prob_density = psi ** 2

        scale_psi = (E1_eV * 0.4) / np.max(np.abs(psi))
        scale_prob = (E1_eV * 0.8) / np.max(prob_density)

        psi_scaled = (psi * scale_psi) + En_eV
        prob_scaled = (prob_density * scale_prob) + En_eV
        color = colors[(n - 1) % len(colors)]

        fig.add_trace(go.Scatter(
            x=[0, L_nm], y=[En_eV, En_eV], mode='lines+markers',
            line=dict(color=color, width=3), marker=dict(size=6, symbol='line-ew-open'),
            name=f"n={n}"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[0, L_nm], y=[En_eV, En_eV], mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', dash='dash'), showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=x, y=psi_scaled, mode='lines',
            line=dict(color=color, width=2, shape='spline'), showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=[0, L_nm], y=[En_eV, En_eV], mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', dash='dash'), showlegend=False
        ), row=1, col=3)
        fig.add_trace(go.Scatter(
            x=x, y=prob_scaled, mode='lines', fill='tonexty',
            fillcolor=hex_to_rgba(color, 0.15),
            line=dict(color=color, width=2, shape='spline'), showlegend=False
        ), row=1, col=3)

    for i in range(1, 4):
        fig.add_vrect(x0=-0.5, x1=0, fillcolor="rgba(100, 100, 100, 0.2)", line_width=0, row=1, col=i)
        fig.add_vrect(x0=L_nm, x1=L_nm + 0.5, fillcolor="rgba(100, 100, 100, 0.2)", line_width=0, row=1, col=i)
        fig.add_vline(x=0, line_width=2, line_color="#ffffff", row=1, col=i)
        fig.add_vline(x=L_nm, line_width=2, line_color="#ffffff", row=1, col=i)
        fig.update_xaxes(title_text="Posição x (nm)", range=[-0.1, L_nm + 0.1], showgrid=False, zeroline=False, row=1, col=i)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, row=1, col=i)

    fig.update_yaxes(title_text="Energia (eV)", range=[0, max_energy * 1.1], row=1, col=1)
    fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, width='stretch')


# ==========================================
# SIMULAÇÃO 2: TUNELAMENTO QUÂNTICO (BARREIRA RETANGULAR)
# ==========================================
elif menu == "Tunelamento":
    with st.sidebar:
        st.markdown("## Parâmetros da Barreira")
        a_nm = st.slider("Largura da Barreira (a em nm):", min_value=0.05, max_value=2.0, value=0.5, step=0.05)
        V0_eV = st.slider("Altura da Barreira (V₀ em eV):", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
        E_eV = st.slider("Energia do Elétron (E em eV):", min_value=0.1, max_value=20.0, value=3.0, step=0.1)

    st.title("Tunelamento Quântico")
    st.markdown(
        "Um elétron descrito por uma onda plana incide sobre uma **barreira de potencial retangular** "
        "de altura $V_0$ e largura $a$. Classicamente, se a energia $E$ do elétron for menor que $V_0$, "
        "ele jamais atravessaria a barreira. Na Mecânica Quântica, porém, a função de onda não vai a zero "
        "dentro da barreira — ela **decai exponencialmente** — e ainda existe uma probabilidade não nula de "
        "encontrar o elétron do outro lado. Esse fenômeno é o **tunelamento quântico**. Use os controles ao lado "
        "para variar a largura da barreira e veja como isso afeta a forma da onda (e a chance de tunelamento)."
    )

    mass_kg = particles["Elétron"]

    # Evita a singularidade exata em E = V0 (ponto de transição entre os regimes)
    E_eV_calc = E_eV
    if abs(E_eV_calc - V0_eV) < 1e-3:
        E_eV_calc = V0_eV - 1e-3

    E_J = E_eV_calc * eV_to_J
    V0_J = V0_eV * eV_to_J
    a_m = a_nm * 1e-9

    # Número de onda nas regiões I e III (fora da barreira)
    k1 = np.sqrt(2 * mass_kg * E_J) / hbar
    # Constante da região II: real -> decaimento (tunelamento); imaginária -> oscilação (acima da barreira)
    kappa = np.sqrt(2 * mass_kg * (V0_J - E_J) + 0j) / hbar

    is_tunneling = E_eV < V0_eV

    # Sistema linear 4x4 a partir da continuidade de psi e psi' em x=0 e x=a
    # Incógnitas: r (reflexão), C, D (coeficientes na barreira), t (transmissão). Amplitude incidente = 1.
    M = np.array([
        [-1, 1, 1, 0],
        [-1j * k1, -kappa, kappa, 0],
        [0, np.exp(kappa * a_m), np.exp(-kappa * a_m), -1],
        [0, kappa * np.exp(kappa * a_m), -kappa * np.exp(-kappa * a_m), -1j * k1]
    ], dtype=complex)
    RHS = np.array([1, -1j * k1, 0, 0], dtype=complex)
    r_amp, C, D, t_amp = np.linalg.solve(M, RHS)

    T_coef = np.abs(t_amp) ** 2
    R_coef = np.abs(r_amp) ** 2

    # Malha espacial para plotagem
    margin_nm = max(1.2, 1.5 * a_nm)
    x_nm = np.linspace(-margin_nm, a_nm + margin_nm, 1600)
    x_m = x_nm * 1e-9

    psi = np.zeros_like(x_m, dtype=complex)
    mask_I = x_m < 0
    mask_II = (x_m >= 0) & (x_m <= a_m)
    mask_III = x_m > a_m

    psi[mask_I] = np.exp(1j * k1 * x_m[mask_I]) + r_amp * np.exp(-1j * k1 * x_m[mask_I])
    psi[mask_II] = C * np.exp(kappa * x_m[mask_II]) + D * np.exp(-kappa * x_m[mask_II])
    psi[mask_III] = t_amp * np.exp(1j * k1 * (x_m[mask_III] - a_m))

    prob_density = np.abs(psi) ** 2

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Massa do Elétron", f"{mass_kg:.2e} kg")
    col2.metric("Energia (E)", f"{E_eV:.1f} eV")
    col3.metric("Altura da Barreira (V₀)", f"{V0_eV:.1f} eV")
    col4.metric("Largura da Barreira (a)", f"{a_nm:.2f} nm")

    col5, col6, col7 = st.columns(3)
    col5.metric("Coeficiente de Transmissão (T)", f"{T_coef * 100:.3f} %")
    col6.metric("Coeficiente de Reflexão (R)", f"{R_coef * 100:.3f} %")
    col7.metric("Regime", "Tunelamento (E < V₀)" if is_tunneling else "Sobre a barreira (E > V₀)")

    st.markdown("---")

    # Potencial V(x) em degrau, para desenhar a barreira junto com a onda (como no diagrama clássico de livro-texto)
    V_plot_eV = np.where((x_nm >= 0) & (x_nm <= a_nm), V0_eV, 0.0)

    # Escala visual (arbitrária) da função de onda, deslocada para "flutuar" na altura de E,
    # exatamente como nos diagramas didáticos de tunelamento.
    amp_ref_psi = max(V0_eV, E_eV) * 0.28
    max_abs_psi = np.max(np.abs(psi.real))
    scale_psi = amp_ref_psi / max_abs_psi if max_abs_psi > 0 else 1.0
    psi_plot = psi.real * scale_psi + E_eV

    amp_ref_prob = max(V0_eV, E_eV) * 0.45
    max_prob = np.max(prob_density)
    scale_prob = amp_ref_prob / max_prob if max_prob > 0 else 1.0
    prob_plot = prob_density * scale_prob + E_eV

    barrier_color = "#ff9100"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Potencial V(x) e Onda Re[ψ(x)]", "Potencial V(x) e Densidade |ψ(x)|²"),
        horizontal_spacing=0.08
    )

    # --- Coluna 1: V(x), E e a parte real da onda ---
    fig.add_vrect(x0=0, x1=a_nm, fillcolor=hex_to_rgba(barrier_color, 0.12), line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_nm, y=V_plot_eV, mode='lines', line=dict(color=barrier_color, width=3, shape='hv'),
        name="V(x)", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[x_nm[0], x_nm[-1]], y=[E_eV, E_eV], mode='lines',
        line=dict(color='rgba(255,255,255,0.6)', width=1.5, dash='dash'),
        name="E", showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_nm, y=psi_plot, mode='lines',
        line=dict(color='#00e5ff', width=2), name="Re[ψ(x)]", showlegend=False
    ), row=1, col=1)

    # --- Coluna 2: V(x), E e a densidade de probabilidade ---
    fig.add_vrect(x0=0, x1=a_nm, fillcolor=hex_to_rgba(barrier_color, 0.12), line_width=0, row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_nm, y=V_plot_eV, mode='lines', line=dict(color=barrier_color, width=3, shape='hv'),
        name="V(x)", showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[x_nm[0], x_nm[-1]], y=[E_eV, E_eV], mode='lines',
        line=dict(color='rgba(255,255,255,0.6)', width=1.5, dash='dash'),
        name="E", showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_nm, y=prob_plot, mode='lines', fill='tonexty',
        fillcolor=hex_to_rgba('#ff00ea', 0.18),
        line=dict(color='#ff00ea', width=2), name="|ψ(x)|²", showlegend=False
    ), row=1, col=2)

    # Rótulos das três regiões, como no diagrama clássico (apenas no painel da onda, para não poluir)
    barrier_label = "decaimento ~e<sup>-κx</sup>" if is_tunneling else "oscila sobre a barreira"
    label_y = E_eV + amp_ref_psi * 1.55
    fig.add_annotation(x=x_nm[0] + margin_nm * 0.15, y=label_y, text="incidente + refletida",
                        showarrow=False, font=dict(color="#00e5ff", size=11), row=1, col=1)
    fig.add_annotation(x=a_nm / 2, y=label_y, text=barrier_label,
                        showarrow=False, font=dict(color=barrier_color, size=11), row=1, col=1)
    fig.add_annotation(x=a_nm + margin_nm * 0.5, y=label_y, text="transmitida",
                        showarrow=False, font=dict(color="#00e5ff", size=11), row=1, col=1)

    for col in (1, 2):
        fig.add_vline(x=0, line_width=1.5, line_color="rgba(255,255,255,0.5)", row=1, col=col)
        fig.add_vline(x=a_nm, line_width=1.5, line_color="rgba(255,255,255,0.5)", row=1, col=col)
        fig.update_xaxes(title_text="Posição x (nm)", range=[x_nm[0], x_nm[-1]],
                          showgrid=False, zeroline=False, row=1, col=col)

    fig.update_yaxes(title_text="V(x), E (eV)  /  Ψ (escala arbitrária)", showgrid=True,
                      gridcolor='rgba(255,255,255,0.1)', zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="V(x), E (eV)  /  |ψ|² (escala arbitrária)", showgrid=True,
                      gridcolor='rgba(255,255,255,0.1)', zeroline=False, row=1, col=2)

    fig.update_layout(height=580, template="plotly_dark", hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "A linha laranja é o potencial V(x) (a barreira) e a linha tracejada branca marca a energia E do elétron — "
        "assim como no diagrama clássico de livro-texto. A curva azul/rosa (ψ) está deslocada para 'flutuar' na "
        "altura de E; sua amplitude é apenas uma escala visual, não literalmente em elétron-volts."
    )

    if is_tunneling:
        st.caption(
            "Repare que, dentro da barreira, a onda deixa de oscilar e decai exponencialmente — é esse decaimento "
            "que sobrevive até o outro lado, dando origem ao tunelamento. Aumente a largura da barreira e veja a "
            "transmissão (T) cair."
        )
    else:
        st.caption(
            "Com E > V₀, o elétron tem energia suficiente para passar 'classicamente' por cima da barreira. "
            "Mesmo assim, note que a onda ainda sofre reflexão parcial (R > 0) dentro da barreira, um efeito "
            "puramente quântico causado pela mudança abrupta de potencial."
        )


# ==========================================
# SIMULAÇÃO 3: ORBITAIS ATÔMICOS 3D
# ==========================================
elif menu == "Orbitais Atômicos 3D":
    st.title("Orbitais Atômicos do Hidrogênio")
    st.markdown(
        "Visualização da densidade de probabilidade tridimensional do elétron no átomo de Hidrogênio "
        "($Z=1$), a solução exata da Equação de Schrödinger. A forma completa do orbital é definida pelos "
        "**três números quânticos**: principal ($n$), azimutal ($l$) e magnético ($m_l$). Varie-os na barra "
        "lateral para explorar todos os formatos possíveis de orbitais."
    )

    with st.sidebar:
        st.markdown("## Números Quânticos")

        n_val = st.slider("Número Quântico Principal (n):", min_value=1, max_value=6, value=3)

        l_dict = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}
        l_max = n_val - 1
        if l_max == 0:
            l_val = 0
            st.caption(f"Regra: 0 ≤ l ≤ n − 1  (para n = {n_val}, o único valor possível é l = 0)")
        else:
            l_val = st.slider("Número Quântico Azimutal (l):", min_value=0, max_value=l_max, value=min(1, l_max))
            st.caption(f"Regra: 0 ≤ l ≤ n − 1  (para n = {n_val}, l vai de 0 a {l_max})")
        st.markdown(f"**Tipo de Orbital:** {n_val}{l_dict.get(l_val, l_val)}")

        if l_val == 0:
            ml_val = 0
            st.markdown("**Número Quântico Magnético (m_l):** 0 (único valor possível para l = 0)")
        else:
            ml_val = st.slider("Número Quântico Magnético (m_l):", min_value=-l_val, max_value=l_val, value=0)

        st.markdown("---")
        st.info(
            "Usamos o átomo de Hidrogênio (Z=1) como referência, pois é o único sistema com solução analítica "
            "exata. Isso permite calcular a função de onda completa ψ(n,l,m_l) = R_nl(r) · Y_l^ml(θ,φ), incluindo "
            "os nós radiais (definidos por n e l) e a forma angular (definida por l e m_l)."
        )

    radial_nodes = n_val - l_val - 1
    angular_nodes = l_val

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric("Orbital Atual", f"{n_val}{l_dict.get(l_val, l_val)} (m_l = {ml_val})")
        st.metric("Nós Radiais (n − l − 1)", radial_nodes)
        st.metric("Nós Angulares (l)", angular_nodes)
        st.markdown(f"""
        ### Efeito dos Números Quânticos
        - **n** controla o **tamanho** do orbital e o número de **nós radiais** (camadas concêntricas de
          probabilidade nula). Quanto maior n, mais "inchado" e mais estratificado fica o orbital.
        - **l** define a **forma geral** (esférica para s, lobular para p, mais complexa para d e f) e o número
          de nós angulares.
        - **m_l** orienta essa forma no espaço em torno do eixo z.

        No gráfico ao lado, as camadas mostradas correspondem a diferentes níveis de densidade de probabilidade
        $|\\psi|^2$ — repare nos "vazios" entre elas quando n − l − 1 > 0.
        """)

    with col2:
        # ==========================================================
        # MODELO 3D DO ORBITAL
        # ==========================================================
        # Construção da superfície em coordenadas esféricas,
        # seguindo a abordagem utilizada no "app":
        # r(theta, phi) ~ |Y_l^m(theta, phi)|²
        #
        # Mantemos n no cálculo para que os diferentes estados
        # radiais continuem representados no modelo.

        resolution_theta = 120
        resolution_phi = 120

        theta = np.linspace(0, np.pi, resolution_theta)
        phi = np.linspace(0, 2 * np.pi, resolution_phi)

        theta, phi = np.meshgrid(theta, phi)

        # Parte angular
        Y_ang = compute_spherical_harmonic(
            l_val,
            ml_val,
            theta,
            phi
        )

        angular_density = np.abs(Y_ang) ** 2

        # ----------------------------------------------------------
        # Parte radial
        # ----------------------------------------------------------
        # Para manter a influência de n e l do app_3 original,
        # calculamos R_nl em uma coordenada radial normalizada.
        #
        # O objetivo aqui é usar a forma radial para determinar
        # a espessura/tamanho do orbital, mantendo a aparência
        # de superfície do "app".

        radial_nodes = n_val - l_val - 1

        # Escala radial característica.
        # O raio aumenta aproximadamente com n².
        radial_scale = n_val ** 2

        # A densidade angular determina a forma do orbital.
        # A contribuição radial é incorporada como um fator de
        # escala para produzir uma superfície visualmente limpa.

        radial_factor = radial_scale * angular_density

        # Normalização para manter os orbitais dentro de uma
        # escala visual semelhante independentemente de n.
        max_radius = np.max(radial_factor)

        if max_radius > 0:
            R = radial_factor / max_radius
        else:
            R = radial_factor

        # Aumenta a escala dos orbitais para que fiquem visualmente
        # semelhantes ao modelo do "app".
        R = R * 0.45

        # Coordenadas cartesianas da superfície polar
        X = R * np.sin(theta) * np.cos(phi)
        Y_coord = R * np.sin(theta) * np.sin(phi)
        Z_coord = R * np.cos(theta)

        # ==========================================================
        # SUPERFÍCIE 3D
        # ==========================================================
        fig3d = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y_coord,
                    z=Z_coord,
                    surfacecolor=angular_density,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(
                        title="Densidade<br>|ψ|²",
                        thickness=15,
                        len=0.6
                    ),
                    cmin=0.0,
                    cmax=np.max(angular_density)
                    if np.max(angular_density) > 0
                    else 1.0,
                    lighting=dict(
                        ambient=0.5,
                        diffuse=0.8,
                        specular=0.5,
                        roughness=0.5
                    )
                )
            ]
        )

        fixed_range = 0.5

        fig3d.update_layout(
            height=600,
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(
                    range=[-fixed_range, fixed_range],
                    showgrid=False,
                    zeroline=True,
                    visible=False
                ),
                yaxis=dict(
                    range=[-fixed_range, fixed_range],
                    showgrid=False,
                    zeroline=True,
                    visible=False
                ),
                zaxis=dict(
                    range=[-fixed_range, fixed_range],
                    showgrid=False,
                    zeroline=True,
                    visible=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='cube'
            )
        )

        st.plotly_chart(fig3d, width='stretch')
