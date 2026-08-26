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
    """Converte uma cor hexadecimal (#rrggbb) para string rgba() com transparência.

    A versão anterior tentava fazer isso com color.replace('rgb', 'rgba')...,
    o que não tem efeito nenhum sobre uma string hexadecimal (o preenchimento
    ficava sólido, sem transparência).
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Calcula Y_l^m(theta, phi) de forma compatível com scipy novo e antigo.

    A partir do SciPy 1.15, `scipy.special.sph_harm` foi removida em favor de
    `sph_harm_y`, que também mudou a ordem dos argumentos (n=l primeiro, depois
    m) — usando sempre theta=ângulo polar [0, pi] e phi=ângulo azimutal
    [0, 2pi]. Sem esse ajuste o app quebra com AttributeError em qualquer
    SciPy recente.
    """
    if hasattr(sp, "sph_harm_y"):
        return sp.sph_harm_y(l, m, theta, phi)
    # Fallback para SciPy < 1.15 (assinatura antiga: sph_harm(m, n, phi, theta))
    return sp.sph_harm(m, l, phi, theta)


# ==========================================
# MENU DE NAVEGAÇÃO
# ==========================================
st.sidebar.image(
    "https://www3.unicentro.br/petfisica/wp-content/uploads/sites/54/2025/05/Dia-da-Mecanica-quantica.jpg",
    width=100
)
st.sidebar.markdown("## Navegação")
menu = st.sidebar.radio("Escolha a simulação:", ["Poço de Potencial Infinito", "Orbitais Atômicos 3D"])
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
    st.markdown("Analise o comportamento ondulatório da matéria sob confinamento 1D.")

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
    # Paleta com 10 cores distintas (antes só havia 8, e n_max pode chegar a 10,
    # o que fazia dois níveis repetirem a mesma cor)
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
            fillcolor=hex_to_rgba(color, 0.15),  # antes o fill ficava sólido (bug de transparência)
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
# SIMULAÇÃO 2: ORBITAIS 3D E ÁTOMOS
# ==========================================
elif menu == "Orbitais Atômicos 3D":
    st.title("Orbitais e Átomos Multieletrônicos")
    st.markdown("Visualização da parte angular da densidade de probabilidade. A forma é definida por $(l, m_l)$, mas o tamanho real do orbital é afetado pelo número atômico ($Z$).")

    with st.sidebar:
        st.markdown("## Escolha o Átomo")
        atomos = {
            "Hidrogênio (H)": 1,
            "Hélio (He)": 2,
            "Lítio (Li)": 3,
            "Carbono (C)": 6,
            "Nitrogênio (N)": 7,
            "Oxigênio (O)": 8,
            "Silício (Si)": 14,
            "Ferro (Fe)": 26
        }
        selected_atom = st.selectbox("Elemento:", list(atomos.keys()))
        Z_atom = atomos[selected_atom]

        st.markdown("---")
        st.markdown("## Números Quânticos")

        l_dict = {0: "s", 1: "p", 2: "d", 3: "f"}
        l_val = st.slider("Número Quântico Azimutal (l):", min_value=0, max_value=3, value=1)
        st.markdown(f"**Tipo de Orbital:** {l_dict[l_val]}")

        # BUG CORRIGIDO: quando l = 0, m_l só pode ser 0. O slider antigo era
        # criado com min_value=-l_val e max_value=l_val, ou seja, 0 e 0 — e o
        # Streamlit lança StreamlitAPIException quando min_value == max_value,
        # o que impedia (com erro) selecionar l = 0. Agora tratamos esse caso
        # sem usar um slider degenerado.
        if l_val == 0:
            ml_val = 0
            st.markdown("**Número Quântico Magnético (m_l):** 0 (único valor possível para l = 0)")
        else:
            ml_val = st.slider("Número Quântico Magnético (m_l):", min_value=-l_val, max_value=l_val, value=0)

        st.markdown("---")
        st.info("Nota: Para átomos além do Hidrogênio, a Equação de Schrödinger requer aproximações (ex: Método Hartree-Fock). Aqui usamos a aproximação hidrogenoide, onde o tamanho do orbital contrai com o aumento de Z.")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric("Número Atômico (Z)", Z_atom)
        st.markdown(f"""
        ### Efeito do Núcleo
        Quanto maior o número de prótons (Z), maior a força de atração eletrostática. Isso faz com que a nuvem eletrônica seja **contraída** para mais perto do núcleo.

        No gráfico ao lado, você verá o orbital encolher à medida que escolhe elementos mais pesados.

        **Orbital atual:** {l_dict[l_val]} (l = {l_val}, m_l = {ml_val})
        """)

    with col2:
        # Criação da malha esférica
        theta = np.linspace(0, np.pi, 100)      # Ângulo polar
        phi = np.linspace(0, 2 * np.pi, 100)    # Ângulo azimutal
        theta, phi = np.meshgrid(theta, phi)

        # Cálculo dos Harmônicos Esféricos
        # BUG CORRIGIDO: scipy.special.sph_harm foi removida em versões
        # recentes do SciPy (>= 1.15), o que fazia essa página inteira quebrar
        # com AttributeError independentemente dos valores de l e m_l
        # escolhidos. compute_spherical_harmonic() usa a nova sph_harm_y
        # quando disponível, com fallback para a API antiga.
        Y = compute_spherical_harmonic(l_val, ml_val, theta, phi)

        # Densidade de Probabilidade Angular
        R = np.abs(Y) ** 2

        if l_val == 0:
            R = R * 5  # Ajuste visual para o orbital 's'

        # APROXIMAÇÃO FÍSICA: Contração do orbital proporcional a 1/√Z
        # Orbitais de átomos mais pesados ficam mais próximos do núcleo
        contraction_factor = 1 / np.sqrt(Z_atom)
        R = R * contraction_factor

        # Conversão para coordenadas cartesianas
        X = R * np.sin(theta) * np.cos(phi)
        Y_coord = R * np.sin(theta) * np.sin(phi)
        Z_coord = R * np.cos(theta)

        # Plotly Surface 3D
        fig3d = go.Figure(data=[go.Surface(
            x=X, y=Y_coord, z=Z_coord,
            surfacecolor=R,
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Densidade<br>|ψ|²", thickness=15, len=0.6),
            cmin=0.0,
            cmax=np.max(R) if np.max(R) > 0 else 1.0,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5, roughness=0.5)
        )])

        # O range é fixo para que o usuário possa ver a contração visualmente ao trocar de átomo
        fixed_range = 0.5

        fig3d.update_layout(
            height=600,
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(range=[-fixed_range, fixed_range], showgrid=False, zeroline=True, visible=False),
                yaxis=dict(range=[-fixed_range, fixed_range], showgrid=False, zeroline=True, visible=False),
                zaxis=dict(range=[-fixed_range, fixed_range], showgrid=False, zeroline=True, visible=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode='cube'
            )
        )

        st.plotly_chart(fig3d, width='stretch')