import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.special as sp
import scipy.linalg as la  # Adicionado para resolver a matriz do poço finito

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


# ==========================================
# MENU DE NAVEGAÇÃO
# ==========================================
st.sidebar.image(
    "https://www3.unicentro.br/petfisica/wp-content/uploads/sites/54/2025/05/Dia-da-Mecanica-quantica.jpg",
    width=100
)
st.sidebar.markdown("## Navegação")
# Adicionando a nova aba de Poço Finito
menu = st.sidebar.radio("Escolha a simulação:", [
    "Poço de Potencial Infinito", 
    "Poço de Potencial Finito", 
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
# SIMULAÇÃO 2: POÇO DE POTENCIAL FINITO
# ==========================================
elif menu == "Poço de Potencial Finito":
    with st.sidebar:
        st.markdown("## Parâmetros do Poço Finito")
        selected_particle = st.selectbox("Elemento/Partícula:", list(particles.keys()))
        mass_kg = particles[selected_particle]

        L_nm = st.slider("Largura do Poço (nm):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        L_m = L_nm * 1e-9

        V0_eV = st.slider("Profundidade do Poço (V₀ em eV):", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        V0_J = V0_eV * eV_to_J

    st.title("Poço de Potencial Finito")
    st.markdown("Ao contrário do poço infinito, as paredes têm uma altura real de energia ($V_0$). Isso permite que o elétron 'vaze' sutilmente para as paredes (tunelamento) e cria um número estritamente **limitado** de estados ligados.")

    # 1. Configuração do Método de Diferenças Finitas (FDM)
    N = 1000 # Número de pontos da malha
    # Estendemos a malha para os lados para podermos visualizar o tunelamento/decaimento
    x_nm = np.linspace(-L_nm, 2 * L_nm, N) 
    x_m = x_nm * 1e-9
    dx_m = x_m[1] - x_m[0]

    # Construção do Potencial V(x)
    V_array_J = np.where((x_nm >= 0) & (x_nm <= L_nm), 0.0, V0_J)

    # Construção da Matriz Hamiltoniana Tridiagonal
    t0 = (hbar ** 2) / (2 * mass_kg * dx_m ** 2)
    diagonal = 2 * t0 + V_array_J
    off_diagonal = -t0 * np.ones(N - 1)

    # Resolução dos Autovalores (Energia) e Autovetores (Funções de Onda)
    E_J, vecs = la.eigh_tridiagonal(diagonal, off_diagonal)
    E_eV = E_J / eV_to_J

    # Filtramos apenas os estados que estão "presos" dentro do poço (E < V0)
    bound_states = [i for i in range(len(E_eV)) if E_eV[i] < V0_eV]

    col1, col2, col3 = st.columns(3)
    col1.metric("Massa da Partícula", f"{mass_kg:.2e} kg")
    col2.metric("Largura do Poço (L)", f"{L_nm:.1f} nm")
    col3.metric("Total de Estados Ligados", len(bound_states))

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

    # Desenhando o fundo do potencial V(x) nos 3 gráficos para contexto visual
    V_plot_eV = np.where((x_nm >= 0) & (x_nm <= L_nm), 0.0, V0_eV)
    
    for i in range(1, 4):
        # Linha branca do formato do poço
        fig.add_trace(go.Scatter(x=x_nm, y=V_plot_eV, mode='lines', line=dict(color='white', width=2), showlegend=False), row=1, col=i)
        # Preenchimento cinza das paredes
        fig.add_vrect(x0=x_nm[0], x1=0, fillcolor="rgba(100, 100, 100, 0.2)", line_width=0, row=1, col=i)
        fig.add_vrect(x0=L_nm, x1=x_nm[-1], fillcolor="rgba(100, 100, 100, 0.2)", line_width=0, row=1, col=i)

    # Plotando os estados ligados (limitando a max 10 para o gráfico não ficar poluído se V0 for gigante)
    max_plot_states = min(len(bound_states), 10)
    for i in range(max_plot_states):
        n = i + 1
        idx = bound_states[i]
        En = E_eV[idx]

        # Extração da função de onda (normalizando com dx)
        psi = vecs[:, idx] / np.sqrt(dx_m)
        prob_density = psi ** 2

        # Escalonamento visual para que as ondas caibam graciosamente dentro das paredes (V0)
        scale_psi = (V0_eV * 0.1) / np.max(np.abs(psi)) if np.max(np.abs(psi)) > 0 else 1
        scale_prob = (V0_eV * 0.2) / np.max(prob_density) if np.max(prob_density) > 0 else 1

        psi_scaled = (psi * scale_psi) + En
        prob_scaled = (prob_density * scale_prob) + En

        color = colors[i % len(colors)]

        # 1. Níveis de Energia
        fig.add_trace(go.Scatter(
            x=[0, L_nm], y=[En, En], mode='lines+markers',
            line=dict(color=color, width=3), marker=dict(size=6, symbol='line-ew-open'),
            name=f"n={n} ({En:.2f} eV)"
        ), row=1, col=1)

        # 2. Função de Onda (Observe o decaimento nas bordas!)
        fig.add_trace(go.Scatter(
            x=[x_nm[0], x_nm[-1]], y=[En, En], mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', dash='dash'), showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=x_nm, y=psi_scaled, mode='lines',
            line=dict(color=color, width=2, shape='spline'), showlegend=False
        ), row=1, col=2)

        # 3. Densidade de Probabilidade
        fig.add_trace(go.Scatter(
            x=[x_nm[0], x_nm[-1]], y=[En, En], mode='lines',
            line=dict(color='rgba(255,255,255,0.2)', dash='dash'), showlegend=False
        ), row=1, col=3)
        fig.add_trace(go.Scatter(
            x=x_nm, y=prob_scaled, mode='lines', fill='tonexty',
            fillcolor=hex_to_rgba(color, 0.15),
            line=dict(color=color, width=2, shape='spline'), showlegend=False
        ), row=1, col=3)

    for i in range(1, 4):
        fig.update_xaxes(title_text="Posição x (nm)", range=[x_nm[0], x_nm[-1]], showgrid=False, zeroline=False, row=1, col=i)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, row=1, col=i)

    fig.update_yaxes(title_text="Energia (eV)", range=[0, V0_eV * 1.1], row=1, col=1)
    fig.update_layout(height=600, template="plotly_dark", hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, width='stretch')


# ==========================================
# SIMULAÇÃO 3: ORBITAIS 3D E ÁTOMOS
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
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        Y = compute_spherical_harmonic(l_val, ml_val, theta, phi)
        R = np.abs(Y) ** 2

        if l_val == 0:
            R = R * 5

        contraction_factor = 1 / np.sqrt(Z_atom)
        R = R * contraction_factor

        X = R * np.sin(theta) * np.cos(phi)
        Y_coord = R * np.sin(theta) * np.sin(phi)
        Z_coord = R * np.cos(theta)

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